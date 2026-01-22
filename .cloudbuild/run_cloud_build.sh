#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}/.."

if [[ "$ENV_FILE" == "" ]]; then
    export ENV_FILE=".env"
fi
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

if [[ "${GOOGLE_CLOUD_PROJECT}" == "" ]]; then
    GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project -q)
fi
if [[ "${GOOGLE_CLOUD_PROJECT}" == "" ]]; then
    echo "ERROR: Run 'gcloud config set project' command to set active project, or set GOOGLE_CLOUD_PROJECT environment variable."
    exit 1
fi

BUILD_SA_NAME="agent-eval-build-sa"
BUILD_SA_EMAIL="${BUILD_SA_NAME}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"
COMMIT_SHORT_HASH=$(git rev-parse --short HEAD)

# Creating service account for build, if it doesn't exist
if ! gcloud iam service-accounts describe "${BUILD_SA_EMAIL}" --project "${GOOGLE_CLOUD_PROJECT}" &> /dev/null; then
    echo "Creating service account ${BUILD_SA_NAME} for Cloud Build."
    gcloud iam service-accounts create ${BUILD_SA_NAME} --project "${GOOGLE_CLOUD_PROJECT}" --display-name "Agent Build Service Account"

    echo "Granting roles to service account ${BUILD_SA_NAME}."
    ROLES=(
        "roles/cloudbuild.builds.builder"
        "roles/run.admin"
        "roles/run.invoker"
        "roles/iam.serviceAccountOpenIdTokenCreator"
        "roles/iam.serviceAccountUser"
        "roles/serviceusage.serviceUsageAdmin"
        "roles/serviceusage.serviceUsageConsumer"
        "roles/aiplatform.user"
    )

    # Loop through and grant each role
    for ROLE in "${ROLES[@]}"; do
        gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
            --member="serviceAccount:$BUILD_SA_EMAIL" \
            --role="$ROLE"
    done
fi

gcloud builds submit --config .cloudbuild/cloudbuild.yaml \
    --service-account="projects/${GOOGLE_CLOUD_PROJECT}/serviceAccounts/${BUILD_SA_EMAIL}" \
    --machine-type=e2-highcpu-32 \
    --timeout=120m \
    --substitutions _COMMIT_SHORT_HASH=$COMMIT_SHORT_HASH,_GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,_GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION,_GOOGLE_CLOUD_REGION=$GOOGLE_CLOUD_REGION
