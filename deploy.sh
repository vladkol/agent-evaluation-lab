#!/bin/bash

set -e

# Deploys the Course Creator multi-agent application to Google Cloud Run.
#
# Parameters:
#   --no-redeploy: (Optional) If set, services that are already deployed and have a URL will not be redeployed.
#   --revision-tag: (Optional) A specific revision tag to apply to the deployment.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"

################## Initialization ##################

# Parse script arguments
NO_REDEPLOY="false" # Redeploying all services by default.
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-redeploy)  NO_REDEPLOY="true"; shift ;;
    --revision-tag) REVISION_TAG="$2"; shift 2 ;;
    *) shift ;; # Ignore unknown flags
  esac
done

# Load .env file if it exists.
# Optionally, use a custom .env file path via ENV_FILE environment variable.
if [[ "$ENV_FILE" == "" ]]; then
    export ENV_FILE=".env"
fi
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# If GOOGLE_CLOUD_PROJECT is not defined, get current project from gcloud CLI
if [[ "${GOOGLE_CLOUD_PROJECT}" == "" ]]; then
    GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project -q)
fi
if [[ "${GOOGLE_CLOUD_PROJECT}" == "" ]]; then
    echo "ERROR: Run 'gcloud config set project' command to set active project, or set GOOGLE_CLOUD_PROJECT environment variable."
    exit 1
fi

# GOOGLE_CLOUD_REGION is the region where Cloud Run services will be deployed.
# GOOGLE_CLOUD_LOCATION is a cloud location used for Gemini API calls, it may be a region, and may be "global".
# If GOOGLE_CLOUD_REGION is not defined, it will be the same as GOOGLE_CLOUD_LOCATION unless GOOGLE_CLOUD_LOCATION is "global".
# In that case, the region will be assigned to the default compute region configured with gcloud CLI.
# If none is configured, "us-central1" is the default value.
if [[ "${GOOGLE_CLOUD_REGION}" == "" ]]; then
    GOOGLE_CLOUD_REGION="${GOOGLE_CLOUD_LOCATION}"
fi
if [[ "${GOOGLE_CLOUD_REGION}" == "global" ]]; then
    echo "GOOGLE_CLOUD_REGION is set to 'global'. Getting a default location for Cloud Run."
    GOOGLE_CLOUD_REGION=""
fi
if [[ "${GOOGLE_CLOUD_REGION}" == "" ]]; then
    GOOGLE_CLOUD_REGION=$(gcloud config get-value compute/region -q)
    if [[ "${GOOGLE_CLOUD_REGION}" == "" ]]; then
        GOOGLE_CLOUD_REGION="us-central1"
        echo "WARNING: Cannot get a configured compute region. Defaulting to ${GOOGLE_CLOUD_REGION}."
    fi
fi
# If GOOGLE_CLOUD_LOCATION is empty, "global" will be used.
if [[ "${GOOGLE_CLOUD_LOCATION}" == "" ]]; then
    GOOGLE_CLOUD_LOCATION="global"
fi

echo "Using project ${GOOGLE_CLOUD_PROJECT}."
echo "Using compute region ${GOOGLE_CLOUD_REGION}."

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION}"
export GOOGLE_CLOUD_REGION="${GOOGLE_CLOUD_REGION}"

################## FUNCTIONS ##################

get_service_url() {
    # Retrieves the url of a Cloud Run service.
    # Parameters:
    #   1. Service Name - name of the service.
    #   2. [Optional] Service Revision tag if not active (serving traffic) revision is needed.
    SERVICE_NAME=$1
    REVISION_TAG_NAME=$2
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $GOOGLE_CLOUD_REGION --project $GOOGLE_CLOUD_PROJECT --format='value(status.url)' 2>/dev/null || echo "")
    if [[ "${SERVICE_URL}" == "" ]]; then
        # No serving service deployment found.
        echo ""
        return 0
    fi
    if [[ "${REVISION_TAG_NAME}" != "" ]]; then
        HAS_REVISON=$(gcloud run services describe $SERVICE_NAME --region $GOOGLE_CLOUD_REGION --project $GOOGLE_CLOUD_PROJECT --format="value(status.traffic.filter(tag='$REVISION_TAG_NAME'))" 2>/dev/null || echo "")
        if [[ "${HAS_REVISON}" == "" ]]; then
            echo ""
            return 0
        fi
        REVISION_TAG_URL_PREFIX="${REVISION_TAG_NAME}---"
        SERVICE_URL="${SERVICE_URL/https:\/\//https://$REVISION_TAG_URL_PREFIX}" # optionally, insert "{tag}---" after "https://"
    fi
    echo $SERVICE_URL
}


deploy_service() {
    # Deploys a Cloud Run service.
    # Parameters:
    #   1. SERVICE_NAME - Name of the service.
    #   2. SOURCE_DIR - Directory containing the source code.
    #   3. ADD_PARAMS - (Optional) Additional gcloud parameters. NOTE: This parameter is not allowed to have newline at the beginning or end of the passed value.
    #   4. REVISION_TAG_NAME - (Optional) Revision tag to apply.
    SERVICE_NAME="$1"
    SOURCE_DIR="$2"
    ADD_PARAMS="$3"
    REVISION_TAG_NAME="$4"

    if [[ "${REVISION_TAG_NAME}" != "" ]]; then
        SERVING_URL=$(get_service_url $SERVICE_NAME 2>/dev/null || echo "")
        # If no existing serving deployment, we cannot use "--no-traffic"
        if [[ "${SERVING_URL}" != "" ]]; then
            TAG_PARAMS=" --no-traffic --tag ${REVISION_TAG} --set-env-vars REVISION_TAG=${REVISION_TAG} "
        else
            TAG_PARAMS=" --tag ${REVISION_TAG} --set-env-vars REVISION_TAG=${REVISION_TAG} "
        fi
    fi

    echo "Deploying ${SERVICE_NAME}..."

    gcloud run deploy $SERVICE_NAME \
        --source "${SOURCE_DIR}" \
        --project $GOOGLE_CLOUD_PROJECT \
        --region $GOOGLE_CLOUD_REGION $TAG_PARAMS \
        --no-allow-unauthenticated $ADD_PARAMS \
        --set-env-vars GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}" \
        --set-env-vars GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION}" \
        --set-env-vars GOOGLE_GENAI_USE_VERTEXAI="true"
}

################## Main Script ##################

# Enable required Google Cloud APIs.
# (Requires serviceusage.serviceUsageAdmin role)
echo "📦 Enabling required Google Cloud APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    serviceusage.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    cloudtrace.googleapis.com \
    --project="${GOOGLE_CLOUD_PROJECT}"

# If not redeployment needed, get current URLs.
if [[ "${NO_REDEPLOY}" == "true" ]]; then
    export RESEARCHER_URL=$(get_service_url "researcher" $REVISION_TAG 2>/dev/null || echo "")
    export CONTENT_BUILDER_URL=$(get_service_url "content-builder" $REVISION_TAG 2>/dev/null || echo "")
    export JUDGE_URL=$(get_service_url "judge" $REVISION_TAG 2>/dev/null || echo "")
    export ORCHESTRATOR_URL=$(get_service_url "orchestrator" $REVISION_TAG 2>/dev/null || echo "")
    export COURSE_CREATOR_URL=$(get_service_url "course-creator" $REVISION_TAG 2>/dev/null || echo "")
fi

if [[ "${RESEARCHER_URL}" == "" ]]; then
    deploy_service researcher agents/researcher "" $REVISION_TAG
    export RESEARCHER_URL=$(get_service_url "researcher" $REVISION_TAG)
fi

if [[ "${CONTENT_BUILDER_URL}" == "" ]]; then
    deploy_service content-builder agents/content_builder "" $REVISION_TAG
    export CONTENT_BUILDER_URL=$(get_service_url "content-builder" $REVISION_TAG)
fi

if [[ "${JUDGE_URL}" == "" ]]; then
    deploy_service judge agents/judge "" $REVISION_TAG
    export JUDGE_URL=$(get_service_url "judge" $REVISION_TAG)
fi

if [[ "${ORCHESTRATOR_URL}" == "" ]]; then
    ADD_VARS="--set-env-vars RESEARCHER_AGENT_CARD_URL=$RESEARCHER_URL/a2a/agent/.well-known/agent-card.json \
        --set-env-vars JUDGE_AGENT_CARD_URL=$JUDGE_URL/a2a/agent/.well-known/agent-card.json \
        --set-env-vars CONTENT_BUILDER_AGENT_CARD_URL=$CONTENT_BUILDER_URL/a2a/agent/.well-known/agent-card.json"
    deploy_service orchestrator agents/orchestrator "$ADD_VARS" $REVISION_TAG
    export ORCHESTRATOR_URL=$(get_service_url "orchestrator" $REVISION_TAG)
fi

if [[ "${COURSE_CREATOR_URL}" == "" ]]; then
    ADD_VARS="--set-env-vars AGENT_SERVER_URL=$ORCHESTRATOR_URL"
    deploy_service course-creator app "$ADD_VARS" $REVISION_TAG

    # Allow unauthenticated access to the web app (ONLY FOR TESTING PURPOSES)
    gcloud run services update course-creator \
        --project $GOOGLE_CLOUD_PROJECT \
        --region $GOOGLE_CLOUD_REGION \
        --no-invoker-iam-check

    export COURSE_CREATOR_URL=$(get_service_url "course-creator" $REVISION_TAG)
fi

echo "🚀 Researcher: ${RESEARCHER_URL}"
echo "🚀 Content Builder: ${CONTENT_BUILDER_URL}"
echo "🚀 Judge: ${JUDGE_URL}"
echo "🚀 Orchestrator: ${ORCHESTRATOR_URL}"
echo "🚀 Course Creator Web App: ${COURSE_CREATOR_URL}"
