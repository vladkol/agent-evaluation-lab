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

from vertexai import types # type: ignore

from dotenv import load_dotenv

from shared.evaluation.evaluate import (
    evaluate_agent,
    get_custom_function_metric
)
from shared.evaluation.tool_metrics import (
    trajectory_precision_func, trajectory_recall_func
)

load_dotenv()

METRIC_THRESHOLD = 0.75
RESEARCHER_URL = os.environ["RESEARCHER_URL"]
ORCHESTRATOR_URL = os.environ["ORCHESTRATOR_URL"]
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

if __name__ == "__main__":
    # TODO: implement evaluation
    pass
