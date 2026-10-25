#!/bin/bash
# Build and deploy the Task Factory cloud gateway (tf-gateway) to Cloud Run in project mockloop-gcs-dadd.
#   bash cloud/deploy.sh build    # Cloud Build -> Artifact Registry (~20-30 min: NDK + four pre-warm builds)
#   bash cloud/deploy.sh deploy   # Cloud Run service from the latest image
set -euo pipefail
P=mockloop-gcs-dadd
R=us-central1
SA=tf-gateway@$P.iam.gserviceaccount.com
IMG=$R-docker.pkg.dev/$P/tf/tf-gateway:latest
cd "$(dirname "$0")"
case "${1:-}" in
  build)
    gcloud builds submit --project $P --region $R --config cloudbuild.yaml . ;;
  deploy)
    # App-level auth: every endpoint but /health needs its own key header (agent, verifier or factory),
    # so the service itself is reachable without a Google identity token.
    gcloud run deploy tf-gateway --project $P --region $R --image $IMG --service-account $SA \
      --cpu 8 --memory 32Gi --timeout 3600 --concurrency 20 --min-instances 1 --max-instances 5 \
      --execution-environment gen2 --no-cpu-throttling --allow-unauthenticated \
      --set-env-vars TF_BUCKET=gs://mockloop-gcs-dadd-devicerun,TF_PREFIX=tf-gateway ;;
  *) echo "usage: $0 build|deploy" >&2; exit 1 ;;
esac
