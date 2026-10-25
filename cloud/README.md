# Cloud gateway (tf-gateway)

The trusted grader, moved off the laptop and the Mac mini into Google Cloud. Agents and verifiers talk to one
HTTPS service; the service builds patches with the Android NDK and times them on Google's Device Run phones.

## Pieces

| Piece | Where |
|---|---|
| Service | Cloud Run `tf-gateway`, region us-central1, project `mockloop-gcs-dadd`, 8 vCPU / 32 GiB, gen2 |
| Image | `us-central1-docker.pkg.dev/mockloop-gcs-dadd/tf/tf-gateway:latest`, built by Cloud Build from `cloud/` |
| Identity | Service account `tf-gateway@mockloop-gcs-dadd.iam.gserviceaccount.com` with `roles/devicerun.admin`, `roles/logging.logWriter`, `roles/serviceusage.serviceUsageConsumer`, and on the bucket `gs://mockloop-gcs-dadd-devicerun` only `roles/storage.objectAdmin` + `roles/storage.legacyBucketReader` |
| Secrets (Secret Manager) | `tf-gateway-hmac`, `tf-gateway-agent-key`, `tf-gateway-verifier-key`, `tf-gateway-factory-key`, and one `tf-task-<task>` per hidden task spec. The service account can read them and add versions to the task specs. |
| State | `gs://mockloop-gcs-dadd-devicerun/tf-gateway/`: `builds/` (tars for Device Run) and `cache/` (calibration results and trusted-build records) |
| Phones | Google Developer Device Platform, Device Run. Default `redfin-30` (Pixel 5, Adreno 620). Weights come from `gs://mockloop-gcs-dadd-devicerun/tf-assets/<port>/weights/`. |

## Endpoints and keys

| Endpoint | Header | Who |
|---|---|---|
| `GET /health` | none | anyone |
| `POST /try {task, patch}` | `X-TF-Agent` | agents (development prompts only) |
| `POST /grade {task, patch}` | `X-TF-Verifier` | the Harbor verifier |
| `POST /calibrate {task, force?}`, `/set_target`, `/measure` | `X-TF-Factory` | the factory |

Local copies of the keys live in `~/.config/task-factory/` (mode 600): `gateway-agent-key`,
`gateway-verifier-key`, `gateway-factory-key`, `gateway-hmac`.

## Rules (same as gateway v2)

- Valid only if, at every generated length, the candidate's tokens exactly match a trusted implementation's
  tokens (`valid_token_sources`, default reference or base) on every prompt; time-vs-length must be linear.
- Reward `clip((speedup - 1 - margin) / (target - 1 - margin), 0, 1)`; base and candidate are timed in the
  same Device Run session on the same phone (paired), so phone-to-phone variation cancels.
- Patches may only touch the task's in-scope files; build files are restored from the commit; builds run as
  an unprivileged user from a `git archive` copy with no git history; keys and hidden specs are held only in
  the server's memory; build logs are capped and scrubbed.
- Infrastructure failures (Device Run, Cloud Storage, git) return `infra_error` and never a reward.

## Deploy

    bash cloud/deploy.sh build     # ~20-30 min
    bash cloud/deploy.sh deploy
    uv run python cloud/gates.py smollm2-image2d-decode --url <service URL>   # calibrate + grade oracle/empty/cheats
    uv run python cloud/reach.py --url <service URL>                          # Antigravity direct-reach test

## Pointing the factory at it

- Factory and agents: `TF_GATEWAY=<service URL>`. `/try` needs `X-TF-Agent`, `/grade` needs `X-TF-Verifier`.
- Antigravity agents can call it directly: store the agent key with `client.credentials.create(id="tf-gateway-agent",
  type="bearer_token", token=..., header_name="X-TF-Agent", prefix="")` and give the sandbox
  `network.allowlist = [{"domain": "<service host>", "credential": "tf-gateway-agent"}]`.
- Harbor verifier: `TF_GATEWAY=<service URL>`, `TF_VERIFIER_KEY` and `TF_GATEWAY_HMAC` as today; the task's
  `allowed_hosts` becomes the service host.
