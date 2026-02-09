# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from urllib.parse import urlparse

import google.auth
from google.auth import impersonated_credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.credentials import Credentials
from google.oauth2.id_token import fetch_id_token_credentials
import httpx


DEFAULT_TIMEOUT = 600.0

async def inject_trace_context(request):
    """Injects trace context into the request headers."""
    if not hasattr(inject_trace_context, "propagator"):
        try:
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
            propagator = TraceContextTextMapPropagator()
        except ImportError:
            propagator = None # type: ignore
        setattr(inject_trace_context, "propagator", propagator)
    else:
        propagator = getattr(inject_trace_context, "propagator")
    if propagator:
        propagator.inject(request.headers)
        if "traceparent" in request.headers:
            # Duplicates traceparent to work around Cloud Run rewriting it.
            request.headers["x-original-traceparent"] = request.headers["traceparent"]

def _get_session_with_id_token(audience: str) -> AuthorizedSession:
    """Returns an authorized session with an ID token for the given audience."""
    credentials, _ = google.auth.default()
    if isinstance(
        credentials,
        impersonated_credentials.Credentials
    ):
        credentials = impersonated_credentials.IDTokenCredentials(
            credentials,
            target_audience=audience,
            include_email=True
        )
        credentials.refresh(Request())
    else:
        try:
            credentials = fetch_id_token_credentials(
                audience,
                Request()
            )
        except (DefaultCredentialsError, TypeError):
            pass
    # If still don't have an id token, try fetching it from gcloud CLI.
    if not hasattr(
        credentials,
        "id_token"
    ) or not credentials.id_token:
        try:
            print("Fetching identity token from gcloud CLI.")
            id_token = subprocess.check_output(
                [
                    "gcloud",
                    "auth",
                    "print-identity-token",
                    "-q"
                ]
            ).decode().strip()
            if id_token:
                refresh_token = subprocess.check_output(
                    [
                        "gcloud",
                        "auth",
                        "print-refresh-token",
                        "-q"
                    ]
                ).decode().strip()
                credentials = Credentials(
                    token=id_token,
                    id_token=id_token,
                    refresh_token=refresh_token
                )
        except subprocess.SubprocessError:
            print("ERROR: Unable to fetch identity token from gcloud CLI.")
    return AuthorizedSession(
        credentials
    )

def create_traced_authenticated_client(
        remote_service_url: str,
        timeout: float = DEFAULT_TIMEOUT
    ) -> httpx.AsyncClient:
    """Creates an httpx.AsyncClient with Google identity token authentication.
    Identity tokens are obtained:
      - If running in Cloud, from Compute Metadata server
      - If running locally, from gcloud CLI

    Args:
        remote_service_url (str): URL of the service to authenticate requests to.
        timeout (float, optional): Request timeout. Defaults to DEFAULT_TIMEOUT.

    Returns:
        httpx.AsyncClient: httpx Client with Google identity token authentication.
    """

    class _IdentityTokenAuth(httpx.Auth):
        def __init__(self, remote_service_url: str):
            parsed_url = urlparse(remote_service_url)
            netloc = parsed_url.netloc
            # For revision-tagged URLs, remove the revision tag prefix
            # e.g. "https://c-2ec6aac---judge-z5re36ny5q-uc.a.run.app" -> "https://judge-z5re36ny5q-uc.a.run.app"
            # (https://docs.cloud.google.com/run/docs/authenticating/service-to-service#acquire-token)
            if "---" in netloc:
                netloc = netloc.split("---", 1)[1] # strip the revision tag
            self.root_url = f"{parsed_url.scheme}://{netloc}"
            self.session = None

        def auth_flow(self, request):
            if self.session:
                id_token = self.session.credentials.token
            else:
                self.session = _get_session_with_id_token(self.root_url)
                id_token = self.session.credentials.token
            request.headers["Authorization"] = f"Bearer {id_token}"
            yield request

    return httpx.AsyncClient(
        auth=_IdentityTokenAuth(remote_service_url),
        follow_redirects=True,
        event_hooks={"request": [inject_trace_context]},
        timeout=timeout,
    )
