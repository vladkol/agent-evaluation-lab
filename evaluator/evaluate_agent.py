# Copyright 2025 Google LLC
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

import asyncio
import json
import os
import time

from vertexai import types # type: ignore

from shared.evaluation.evaluate import evaluate_agent
from shared.evaluation.tool_metrics import (
    trajectory_precision, trajectory_recall
)

METRIC_THRESHOLD = 0.8
RESEARCHER_URL = os.environ["RESEARCHER_URL"]
ORCHESTRATOR_URL = os.environ["ORCHESTRATOR_URL"]
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
COMMIT_REVISION_TAG = os.getenv("COMMIT_REVISION_TAG", "latest")

if __name__ == "__main__":
    eval_data_researcher = os.path.dirname(__file__) + "/eval_data_researcher.json"
    metrics=[
        types.RubricMetric.FINAL_RESPONSE_MATCH,
        types.RubricMetric.FINAL_RESPONSE_QUALITY,
        types.RubricMetric.TOOL_USE_QUALITY,
        trajectory_precision,
        trajectory_recall
    ]
    eval_results = asyncio.run(evaluate_agent(
        agent_api_server=RESEARCHER_URL,
        agent_name="agent",
        evaluation_data_file=eval_data_researcher,
        evaluation_storage_uri=f"gs://{GOOGLE_CLOUD_PROJECT}-agents/evaluation",
        metrics=metrics,
        experiment_name="agent-evaluation-researcher",
        run_name=f"{COMMIT_REVISION_TAG}--{int(time.time())}",
    ))
    researcher_eval_failed = False
    print(f"\n🧪 Researcher Evaluation results:\n{json.dumps(eval_results, indent=2)}")
    for metric_name, metric_values in eval_results.items():
        if metric_values["mean"] < METRIC_THRESHOLD:
            print(f"🛑 Researcher Evaluation failed with metric `{metric_name}` below {METRIC_THRESHOLD} threshold.")
            researcher_eval_failed = True
    if not researcher_eval_failed:
        print(f"✅ Researcher Evaluation passed.")

    eval_data_orchestrator = os.path.dirname(__file__) + "/eval_data_orchestrator.json"
    metrics=[
        # types.RubricMetric.FINAL_RESPONSE_QUALITY,
        types.RubricMetric.HALLUCINATION,
    ]
    eval_results = asyncio.run(evaluate_agent(
        agent_api_server=ORCHESTRATOR_URL,
        agent_name="agent",
        evaluation_data_file=eval_data_orchestrator,
        evaluation_storage_uri=f"gs://{GOOGLE_CLOUD_PROJECT}-agents/evaluation",
        metrics=metrics,
        experiment_name="agent-evaluation-orchestrator",
        run_name=f"{COMMIT_REVISION_TAG}--{int(time.time())}",
    ))
    orchestrator_eval_failed = False
    print(f"\n🧪 Orchestrator Evaluation results:\n{json.dumps(eval_results, indent=2)}")
    for metric_name, metric_values in eval_results.items():
        if metric_values["mean"] < METRIC_THRESHOLD:
            print(f"🛑 Orchestrator Evaluation failed with metric `{metric_name}` below {METRIC_THRESHOLD} threshold.")
            orchestrator_eval_failed = True
    if not orchestrator_eval_failed:
        print(f"✅ Orchestrator Evaluation passed.")

    if researcher_eval_failed or orchestrator_eval_failed:
        exit(1)
    else:
        print(f"🎉 All Agent Evaluations passed.")
