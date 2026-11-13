#!/usr/bin/env bash
# Unattended runner for the Glow UI revamp.
#
#   tmux new -s glow
#   ./tools/run_revamp.sh            # uses sonnet by default
#   MODEL=opus ./tools/run_revamp.sh # override model
#
# Loop: run one headless Claude Code pass → read its final line:
#   WAITING_FOR_REVIEW → poll REVIEW.md every 10 min until Decision != pending, then run again
#   CONTINUE           → run again immediately
#   DONE               → exit 0
#   anything else / non-zero exit (e.g. usage limit) → sleep 30 min, run again
#
# The agent's state lives in PROGRESS.md / REVIEW.md / git, so every pass starts a fresh session.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
LOG="$REPO/revamp.log"
MODEL="${MODEL:-sonnet}"
PROMPT_FILE="$REPO/tools/revamp_prompt.md"
POLL_SECS="${POLL_SECS:-600}"
RETRY_SECS="${RETRY_SECS:-1800}"

if ! command -v claude >/dev/null; then echo "claude CLI not found in PATH"; exit 1; fi
if [ "$(git branch --show-current)" != "revamp" ]; then
  echo "Not on branch 'revamp' (on $(git branch --show-current)). Refusing to run."; exit 1
fi

log() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

decision() { grep -m1 '^Decision:' "$REPO/REVIEW.md" | sed 's/^Decision:[[:space:]]*//'; }

log "runner start (model=$MODEL)"
while true; do
  log "--- pass start"
  out=$(claude -p "$(cat "$PROMPT_FILE")" \
          --model "$MODEL" \
          --permission-mode acceptEdits \
          2>&1 | tee -a "$LOG")
  rc=$?
  last=$(printf '%s\n' "$out" | grep -E '^(WAITING_FOR_REVIEW|CONTINUE|DONE)$' | tail -1)
  log "--- pass end rc=$rc signal=${last:-none}"

  case "$last" in
    DONE)
      log "all items complete"; exit 0 ;;
    CONTINUE)
      sleep 5 ;;
    WAITING_FOR_REVIEW)
      log "waiting for review: edit REVIEW.md (Decision: approved | changes: ... | skip)"
      while [ "$(decision)" = "pending" ]; do sleep "$POLL_SECS"; done
      log "review decision: $(decision)" ;;
    *)
      log "no signal (usage limit or error?) — retry in $((RETRY_SECS/60)) min"
      sleep "$RETRY_SECS" ;;
  esac
done
