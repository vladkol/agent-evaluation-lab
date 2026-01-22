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
"""
Evaluation module for Agent Runtime.

This module provides functionality to evaluate agents by running inference against
a dataset of prompts and calculating metrics using Vertex AI Evaluation.
"""

import asyncio
import io
import json
from pathlib import Path
import os
import sys
import tempfile
from typing import Any, Dict,List, Optional

from dotenv import load_dotenv

from httpx import AsyncClient, Limits, HTTPStatusError
from httpx_sse import aconnect_sse

from google.api_core.exceptions import AlreadyExists
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import execution
from google.cloud.exceptions import Conflict
from google.cloud.storage import Blob, Client as StorageClient
from google.genai import types as genai_types

import pandas as pd

from vertexai import init, types, Client # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from authenticated_httpx import create_authenticated_client # type: ignore


MAX_AGENT_REQUESTS = 10
REQUEST_TIMEOUT = 300.0
USER_ID = "evaluation_user"
DEFAULT_REGION = "us-central1"

async def evaluate_agent(
    agent_api_server: str,
    agent_name: str,
    evaluation_data_file: str,
    evaluation_storage_uri: str,
    metrics: List[types.Metric],
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates an agent against a given dataset.

    This function performs the following steps:
    1.  Reads evaluation data from a file (local or GCS).
    2.  Runs parallel inference against the agent API.
    3.  Uploads the inference results and original data to GCS.
    4.  Runs Vertex AI Evaluation using the provided metrics.
    5.  Logs results to Vertex AI Experiment (if configured).

    Args:
        agent_api_server: Base URL of the agent's API server.
        agent_name: Name of the agent service.
        evaluation_data_file: Path or URI to the evaluation dataset (JSON, JSONL, CSV, Parquet).
        evaluation_storage_uri: GCS URI for storing evaluation artifacts.
        metrics: List of Vertex AI metrics to calculate.
        experiment_name: Optional name of the Vertex AI experiment to log to.
        run_name: Optional name of the Vertex AI run to log to.

    Returns:
        A dictionary mapping metric names to their mean and standard deviation scores (as "mean" and "stdev" keys).
    """
    # Load environment variables and set up project/location

    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv(
        "GOOGLE_CLOUD_REGION",
        os.getenv("GOOGLE_CLOUD_LOCATION", DEFAULT_REGION)
    )
    if location == "global":
        location = DEFAULT_REGION
    if not project_id:
        raise ValueError("Project ID not found. Please set GOOGLE_CLOUD_PROJECT environment variable.")
    if not location:
        raise ValueError("Location not found. Please set GOOGLE_CLOUD_REGION or GOOGLE_CLOUD_LOCATION environment variable.")
    init(
        project=project_id,
        location=location,
        experiment=experiment_name,
        experiment_description="Agent Evaluation"
    )
    client = Client( # type: ignore
        project=project_id,
        location=location,
        http_options=genai_types.HttpOptions(api_version="v1beta1"),
    )

    print(f"Starting agent evaluation for `{agent_name}` at {agent_api_server}.")

    # Read evaluation data from GCS or local file
    if evaluation_data_file.startswith("gs://"):
        evaluation_data = Blob.from_uri(evaluation_data_file).download_as_bytes()
    else:
        evaluation_data = Path(evaluation_data_file).read_bytes()

    with io.BytesIO(evaluation_data) as evaluation_data_io:
        data_file_lower = evaluation_data_file.lower()
        if data_file_lower.endswith(".json"):
            eval_data_df = pd.read_json(evaluation_data_io, orient="columns")
        elif data_file_lower.endswith(".jsonl"):
            eval_data_df = pd.read_json(evaluation_data_io, lines=True)
        elif data_file_lower.endswith(".csv"):
            eval_data_df = pd.read_csv(evaluation_data_io)
        elif data_file_lower.endswith(".parquet"):
            eval_data_df = pd.read_parquet(evaluation_data_io)
        else:
            raise ValueError(f"Unsupported file format: {evaluation_data_file}")

    # Prepare session inputs for the agent
    session_inputs = types.evals.SessionInput(
            user_id=USER_ID,
            app_name=agent_name,
            state={},
        )
    eval_data_df["session_inputs"] = [session_inputs] * len(eval_data_df)

    # Initialize authenticated HTTP client
    tmp_httpx_client = create_authenticated_client(
        agent_api_server
    )
    httpx_client = AsyncClient(
        auth=tmp_httpx_client.auth,
        timeout=REQUEST_TIMEOUT,
        limits = Limits(
            max_connections=MAX_AGENT_REQUESTS * 2,
            max_keepalive_connections=MAX_AGENT_REQUESTS,
            keepalive_expiry=REQUEST_TIMEOUT
        )
    )
    tmp_httpx_client = None

    # Trying to get agent info from
    agent_info_url = f"{agent_api_server}/apps/{agent_name}/agent-info"
    agent_info_response = await httpx_client.get(agent_info_url)
    if not agent_info_response.is_error:
        agent_info = types.evals.AgentInfo.model_validate_json(
            agent_info_response.content.decode("utf-8")
        )
    else:
        agent_info = types.evals.AgentInfo(
            name=agent_name
        )

    # Ensure the GCS bucket for evaluation artifacts exists
    evaluation_storage_uri = evaluation_storage_uri.rstrip("/")
    storage_client = StorageClient(project=project_id)
    try:
        artifacts_bucket = evaluation_storage_uri.replace("gs://", "").split(
            "/", 1
        )[0]
        storage_client.create_bucket(artifacts_bucket, location=location)
    except Conflict:
        pass

    # Run inference in parallel to generate agent responses
    print("Running agent inference...")
    eval_df_with_inference = await run_parallel_inference(
        httpx_client,
        agent_api_server,
        agent_name,
        USER_ID,
        eval_data_df
    )

    # Upload the dataset with inference results to GCS
    time_str = str(pd.Timestamp.now().value)
    eval_data_gcs_data_location = f"{evaluation_storage_uri}/evaluation_dataset/{time_str}.jsonl"
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        eval_df_with_inference.to_json(tmp_file, orient="records", lines=True)
        tmp_file.flush()
        Blob.from_uri(
            eval_data_gcs_data_location,
            client=storage_client
        ).upload_from_filename(
            filename=tmp_file.name,
            client=storage_client,
            content_type="application/json"
        )
    agent_dataset_with_inference = types.EvaluationDataset(
        gcs_source = types.GcsSource(uris=[eval_data_gcs_data_location])
    )

    # Run Vertex AI Evaluation
    run = None
    try:
        if experiment_name and run_name:
            try:
                run = aiplatform.start_run(run=run_name)
            except AlreadyExists:
                run = aiplatform.start_run(run=run_name, resume=True)
            experiment_subfolder = f"/{experiment_name}/{run_name}/"
        else:
            experiment_subfolder = "/"
        evaluation_run = client.evals.evaluate(
            dataset=agent_dataset_with_inference,
            agent_info = agent_info,
            metrics=metrics,
        )
        summary_metrics = {
            m.metric_name : {
                "mean": m.mean_score or 0.0,
                "stdev": m.stdev_score or 0.0
            }
            for m in evaluation_run.summary_metrics
        }
        json_data = evaluation_run.model_dump_json(indent=2, exclude_none=True)
        eval_result_dest = f"{evaluation_storage_uri}/evaluation_results{experiment_subfolder}eval_results.json"
        Blob.from_uri(eval_result_dest, client=storage_client).upload_from_string(
            data=json_data,
            client=storage_client,
            content_type="application/json"
        )
        metrics_to_log = {}
        for metric_name, metric_value in summary_metrics.items():
            metrics_to_log[f"{metric_name}/mean"] = metric_value["mean"] or 0.0
            metrics_to_log[f"{metric_name}/stdev"] = metric_value["stdev"] or 0.0
        if run:
            run.log_metrics(metrics_to_log)
            run.log_params({
                "evaluation_results": eval_result_dest,
                "evaluation_data": eval_data_gcs_data_location
            })
            run_id = run.resource_id.rsplit("/", 1)[-1]
            console_url = f"https://console.cloud.google.com/vertex-ai/experiments/locations/{location}/experiments/{experiment_name}/runs/{run_id}/metrics?project={project_id}"
            print(f"Evaluation run: {console_url}")
        return summary_metrics
    except:
        if run:
            run = None
            aiplatform.end_run(execution.Execution.State.FAILED)
        raise
    finally:
        if run:
            aiplatform.end_run(execution.Execution.State.COMPLETE)


async def _create_session(
        httpx_client: AsyncClient,
        agent_server_origin: str,
        agent_name: str,
        user_id: str
) -> str:
    """
    Creates a new session for the user on the agent server.

    Args:
        httpx_client: Authenticated AsyncClient for making requests.
        agent_server_origin: Base URL of the agent server.
        agent_name: Name of the agent application.
        user_id: ID of the user.

    Returns:
        The session ID of the newly created session.
    """
    content_headers = [
        ("Content-Type", "application/json"),
    ]
    agent_server_origin = agent_server_origin.strip("/")
    session_request_url = f"{agent_server_origin}/apps/{agent_name}/users/{user_id}/sessions"
    session_response = await httpx_client.post(
        session_request_url,
        headers=content_headers
    )
    session_response.raise_for_status()
    session_id = session_response.json()["id"]
    return session_id

async def _run_inference(
        semaphore: asyncio.Semaphore,
        httpx_client: AsyncClient,
        agent_server_origin: str,
        agent_name: str,
        user_id: str,
        prompt: str
) -> List[Dict[str, Any]]:
    """
    Runs inference for a single prompt, handling session management and SSE stream.

    Args:
        semaphore: Semaphore to limit concurrency.
        httpx_client: Authenticated AsyncClient.
        agent_server_origin: Base URL of the agent server.
        agent_name: Name of the agent application.
        user_id: User ID.
        prompt: The input prompt text.

    Returns:
        A list of event dictionaries received from the agent's SSE stream.
    """
    async with semaphore:
        agent_server_origin = agent_server_origin.strip("/")
        session_id = await _create_session(
            httpx_client,
            agent_server_origin,
            agent_name,
            user_id
        )

        request = {
            "appName": agent_name,
            "userId": user_id,
            "sessionId": session_id,
            "newMessage": {
                "role": "user",
                "parts": [{"text": prompt}]
            },
            "streaming": False
        }
        make_another_request = True
        events = []
        while make_another_request:
            make_another_request = False
            async with aconnect_sse(
                httpx_client,
                "POST",
                f"{agent_server_origin}/run_sse",
                json=request
            ) as event_source:
                if event_source.response.is_error:
                    await event_source.response.aread()
                    try:
                        event_source.response.raise_for_status()
                    except HTTPStatusError as e:
                        if (
                            e.response.status_code == 404
                            and "session" in event_source.response.text.lower()
                        ):
                            print("Session not found. Trying with another one.")
                            session_id = await _create_session(
                                httpx_client,
                                agent_server_origin,
                                agent_name,
                                user_id
                            )
                            request["sessionId"] = session_id
                            make_another_request = True
                        else:
                            events.append(
                                {
                                    "content":{
                                        "parts": [
                                            {
                                                "text": f"Error {event_source.response.text}"
                                            }
                                        ]
                                    }
                                }
                            )
                else:
                    async for server_event in event_source.aiter_sse():
                        try:
                            event = server_event.json()
                            events.append(event)
                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON from SSE event: {server_event.data}")
                            continue
        return events

async def run_parallel_inference(
        httpx_client: AsyncClient,
        agent_server_origin: str,
        agent_name: str,
        user_id: str,
        eval_dataset_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Runs inference in parallel for all prompts in the evaluation dataset.

    Args:
        httpx_client: Authenticated AsyncClient.
        agent_server_origin: Base URL of the agent server.
        agent_name: Name of the agent application.
        user_id: User ID.
        eval_dataset_df: DataFrame containing the evaluation dataset (must have a "prompt" column).

    Returns:
        A new DataFrame containing the original data plus "response" and "intermediate_events" columns.
    """
    prompts = eval_dataset_df["prompt"].to_list()
    sem = asyncio.Semaphore(MAX_AGENT_REQUESTS)
    tasks = [
        _run_inference(
            sem,
            httpx_client,
            agent_server_origin,
            agent_name,
            user_id,
            prompt,
        )
        for prompt in prompts
    ]
    responses = await asyncio.gather(*tasks)
    responses_df = _process_agent_responses(responses)
    combined_df = eval_dataset_df.join(responses_df)
    return combined_df

def _get_response_text(response: Dict[str, Any]) -> str:
    """
    Extracts text from text parts a response event.

    Args:
        response: Response event dictionary.

    Returns:
        The final response text.
    """
    content = response.get("content", {})
    if (
        not content
        or not content.get("parts")
        or content.get("role", "") not in ["model", ""]
    ):
        return ""
    texts = []
    for part in content["parts"]:
        if part.get("text"):
            texts.append(part["text"])
    return " ".join(texts)

def _process_agent_responses(
    agent_responses: List[List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Processes raw agent responses into a structured format for the DataFrame.

    Extracts the final response text and organizes intermediate events.

    Args:
        agent_responses: List of lists, where each inner list contains event dictionaries for a single run.

    Returns:
        A DataFrame with "response" (final text) and "intermediate_events" (list of dicts) columns.
    """
    processed_intermediate_events = []
    processed_responses = []
    for response_list in agent_responses:
        intermediate_events_row = []
        response_row = ""
        try:
            # Remove trailing responses without text content
            while (
                response_list
                and _get_response_text(response_list[-1]).strip() == ""
            ):
                response_list.pop()
            if response_list:
                response_row = _get_response_text(response_list[-1])
                for intermediate_event in response_list[:-1]:
                    if not intermediate_event.get("content"):
                        continue
                    intermediate_events_row.append(
                        {
                            "event_id": intermediate_event["id"],
                            "content": intermediate_event["content"],
                            "creation_timestamp": intermediate_event["timestamp"],
                            "author": intermediate_event["author"],
                        }
                    )
        except Exception as e:  # pylint: disable=broad-exception-caught
            error_payload = {
                "error": (
                    f"Failed to parse agent run response {str(response_list)} to "
                    f"intermediate events and final response: {e}"
                ),
            }
            response_row = json.dumps(error_payload)
        processed_intermediate_events.append(intermediate_events_row)
        processed_responses.append(response_row)
    return pd.DataFrame(
        {
            "response": processed_responses,
            "intermediate_events": processed_intermediate_events
        }
    )
