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

METRIC_THRESHOLD = 0.75
RESEARCHER_URL = os.environ["RESEARCHER_URL"]
ORCHESTRATOR_URL = os.environ["ORCHESTRATOR_URL"]
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
COMMIT_REVISION_TAG = os.getenv("COMMIT_REVISION_TAG", "latest")

if __name__ == "__main__":
    # TODO: implement evaluation
    pass
