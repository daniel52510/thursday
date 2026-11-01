#!/bin/sh
# THURSDAY's deployer: waits for runner-out/deploy/request.json (written by core when Daniel taps
# Deploy on a merged change), then:
#   1. refuses if the checkout has uncommitted changes to tracked files (the build would ship them)
#   2. runs the test suite in the sandbox on the committed code; refuses if it fails
#   3. tags the running image thursday-core:previous, rebuilds and restarts thursday-core
#   4. health: the web UI answers, the spoken fillers render, no traceback in the logs (3 min)
#   5. not healthy: puts thursday-core:previous back and restarts that
# Each outcome goes to runner-out/deploy/status.json; core pushes it to his phone. It never pushes
# to GitHub and never changes his checkout. The repo is mounted at its host path (compose needs it).
set -u
cd "$REPO"
D=runner-out/deploy
mkdir -p "$D"

status() {  # state, detail
  printf '{"state": "%s", "detail": "%s", "job": "%s", "at": "%s"}\n' "$1" "$2" "${JOB:-}" "$(date -u +%FT%TZ)" > "$D/status.tmp"
  mv "$D/status.tmp" "$D/status.json"
  echo "deploy ${JOB:-}: $1 $2"
}

healthy() {  # since
  i=0
  while [ $i -lt 90 ]; do
    if curl -fs -o /dev/null -m 3 http://thursday-core:8000/ && docker logs --since "$1" thursday-core 2>&1 | grep -q "spoken fillers ready"; then
      ! docker logs --since "$1" thursday-core 2>&1 | grep -q "Traceback"
      return $?
    fi
    i=$((i + 1)); sleep 2
  done
  return 1
}

echo "deployer ready"
while true; do
  if [ -f "$D/request.json" ]; then
    JOB=$(sed -n 's/.*"job": *"\([^"]*\)".*/\1/p' "$D/request.json")
    rm -f "$D/request.json"
    git config --global --add safe.directory "$REPO" 2>/dev/null
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
      status refused "uncommitted changes in the checkout: commit or stash them, then deploy"
    else
      status testing "running the tests on $(git rev-parse --short HEAD)"
      if ! out=$(docker compose run --rm --entrypoint sh runner -c \
          'git clone -q --branch "$RUNNER_BASE" /repo.git /tmp/w && cd /tmp/w && python -m pytest -q -p no:cacheprovider tests' 2>&1); then
        status refused "tests failed: $(echo "$out" | grep -E 'passed|failed|error' | tail -1 | tr -d '"')"
      else
        status building "tests pass; rebuilding thursday-core"
        docker image tag thursday-core:latest thursday-core:previous 2>/dev/null
        since=$(date -u +%FT%TZ)
        if docker compose up -d --build --no-deps thursday-core >/dev/null 2>&1 && healthy "$since"; then
          status live "$(git rev-parse --short HEAD) is live and healthy"
        else
          docker image tag thursday-core:previous thursday-core:latest
          docker compose up -d --no-deps thursday-core >/dev/null 2>&1
          if healthy "$(date -u +%FT%TZ)"; then
            status rolled_back "the new build wasn't healthy; the previous one is back"
          else
            status broken "the new build wasn't healthy and the previous one didn't come back either: check docker logs thursday-core"
          fi
        fi
      fi
    fi
  fi
  sleep 3
done
