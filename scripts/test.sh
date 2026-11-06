#!/usr/bin/env bash
# Runs tests/ in the runner image (it already has requirements + pytest): repo mounted read-only, no network.
#   scripts/test.sh [pytest args]     e.g. scripts/test.sh tests/test_telephony.py -x
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker run --rm --network none --entrypoint python -v "$PWD":/src:ro -w /src \
  -e PYTHONDONTWRITEBYTECODE=1 thursday-runner:latest -m pytest -q -p no:cacheprovider "${@:-tests}"
