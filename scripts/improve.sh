#!/usr/bin/env bash
# One self-improvement job: THURSDAY's runner (runner/agent.py) works in its sandbox, and a
# passing change comes back as a local branch thursday/improve-<id> for Daniel to review and push.
# Nothing is merged, pushed or deployed. The runner sees only committed code on RUNNER_BASE.
#
#   scripts/improve.sh "Make weather answers one sentence shorter"
set -euo pipefail
cd "$(dirname "$0")/.."
task=${1:?usage: scripts/improve.sh "<task>"}
id=$(date +%Y%m%d-%H%M%S)
mkdir -p runner-out

docker compose --profile runner run --rm -e JOB_ID="$id" runner "$task"

report=runner-out/$id.json
[ -f "$report" ] || { echo "no report: the job crashed (see output above)" >&2; exit 1; }
status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$report")
branch=thursday/improve-$id
if [ "$status" = ready ]; then
  git fetch -q "runner-out/$id.bundle" "$branch:$branch"
  base=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base"])' "$report")
  echo; echo "== $branch (vs $base)"; git --no-pager diff --stat "$base...$branch"
  echo; echo "review and approve: Agentic page, Changes to review (or: git diff $base...$branch)"
  title="THURSDAY has a change for you to review"
else
  title="THURSDAY's improvement job ended: $status"
fi
summary=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("summary") or r.get("error") or "")' "$report")
echo; echo "$summary"

# Phone (ntfy, as approvals.push does): topic in the body, never printed.
topic=$(grep -E '^NTFY_TOPIC=' .env 2>/dev/null | cut -d= -f2- || true)
if [ -n "$topic" ]; then
  python3 - "$topic" "$title" "$branch: $summary" <<'EOF' >/dev/null 2>&1 || echo "(phone push failed)"
import json, sys, urllib.request
topic, title, message = sys.argv[1:4]
body = json.dumps({"topic": topic, "title": title, "message": message[:3000], "tags": ["hammer_and_wrench"],
                   "click": "http://thursday.local/#agentic"}).encode()  # the Agentic page's review cards
urllib.request.urlopen(urllib.request.Request("http://localhost:2586", body, {"Content-Type": "application/json"}), timeout=5)
EOF
fi
