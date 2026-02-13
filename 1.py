import logging
import os
import subprocess
from urllib.parse import urlparse

import google.auth
from google.auth.transport.requests import AuthorizedSession, Request
from google.auth.exceptions import DefaultCredentialsError
from google.auth import impersonated_credentials
from google.auth import compute_engine
from google.oauth2.credentials import Credentials
from google.oauth2.id_token import fetch_id_token_credentials
import httpx


DEFAULT_TIMEOUT = 600.0
logger = logging.getLogger(__name__)

class MakeCall:
    def __init__(self):
        self.creds = None
        self.project = None
        self._default_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        self.impersonate_sa = None

    def make_call(self, url: str, method: str = "GET", headers: dict = None, data: dict = None):
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc
        # For revision-tagged URLs, remove the revision tag prefix
        # e.g. "https://c-2ec6aac---judge-z5re36ny5q-uc.a.run.app" -> "https://judge-z5re36ny5q-uc.a.run.app"
        # (https://docs.cloud.google.com/run/docs/authenticating/service-to-service#acquire-token)
        if "---" in netloc:
            netloc = netloc.split("---", 1)[1] # strip the revision tag
        aud = f"{parsed_url.scheme}://{netloc}"
        creds = self._get_creds(aud)
        print(creds.to_json())
        with httpx.Client() as client:
            headers = headers or {}
            headers["Authorization"] = f"Bearer {creds.token}"
            response = client.request(method, url, headers=headers, data=data)
            response.raise_for_status()
            return response.json()

    def _get_creds(self, aud: str = None) -> Credentials:
            """
            Lazily loads and returns the appropriate credentials for a request.

            This method ensures base credentials (ADC) are loaded only once. It then
            returns the correct credential object—either the base one or a wrapped
            version for ID token generation—based on the provided audience.
            """
            # Step 1: Lazily load the base Application Default Credentials if they don't exist.
            if self.creds is None:
                logger.info("[_get_creds] First-time credential initialization.")
                self.creds, self.project = google.auth.default(scopes=self._default_scopes)
                logger.info(f"Base credentials loaded. Type: {type(self.creds)}, Project: {self.project}")


            default_creds = self.creds


            # Step 2: If no audience is provided, the base credentials are sufficient.
            if not aud:
                logger.info(f"No audience provided, using base credentials. Type: {type(default_creds)}")
                return default_creds


            # Step 3: If an audience is provided, wrap the base credentials to enable ID token generation.
            # This unified logic is more robust than the previous direct fetch.
            if isinstance(default_creds, impersonated_credentials.Credentials):
                logger.info(f"Wrapping impersonated creds for ID token with aud: {aud}")
                return impersonated_credentials.IDTokenCredentials(default_creds, target_audience=aud, include_email=True)


            if self.impersonate_sa:
                target_creds = impersonated_credentials.Credentials(
                    source_credentials=default_creds,
                    target_principal=self.impersonate_sa,
                    delegates=[],
                    target_scopes=self._default_scopes,
                )
                return impersonated_credentials.IDTokenCredentials(target_creds, target_audience=aud, include_email=True)


            if isinstance(default_creds, compute_engine.Credentials):
                logger.info(f"Wrapping compute engine creds for ID token with aud: {aud}")
                return compute_engine.IDTokenCredentials(self.auth_req, target_audience=aud)


            # Fallback for other credential types that may not support ID tokens.
            logger.warning(f"Audience [{aud}] provided, credential type [{type(default_creds)}] "
                        f"may not support ID token wrapping. Falling back to access token.")
            return default_creds

MakeCall().make_call("https://researcher-408387983453.us-central1.run.app")