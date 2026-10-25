#!/bin/bash
# Hidden verifier. Runs in a separate container that only receives /workspace/submission. It sends the
# submitted patch to the device gateway's /grade, checks the gateway's signature, and writes the reward.
# No agent code runs here: the patch is built and run by the gateway on the Mac mini and the phone.
mkdir -p /logs/verifier
python3 /tests/grade_client.py /workspace/submission/patch.diff
