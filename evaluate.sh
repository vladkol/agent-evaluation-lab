#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"

export COMMIT_SHORT_HASH=$(git rev-parse --short HEAD)
export COMMIT_REVISION_TAG="c-${COMMIT_SHORT_HASH}"

echo "➡️ Deployment for evaluation:"
echo "  * Commit: ${COMMIT_SHORT_HASH}"
echo "  * Revision tag: ${COMMIT_REVISION_TAG}"

# Deploy services with a revision tag.
source ./deploy.sh --revision-tag $COMMIT_REVISION_TAG --no-redeploy

echo "🧪 Running evaluation"
uv run -m evaluator.evaluate_agent
