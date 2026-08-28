#!/usr/bin/env bash
# Wrapper for parallel make jobs.
#
# Usage: runone.sh <job-name> <log-dir> <command> [args...]
#
# Runs <command> with stdout/stderr redirected to <log-dir>/<job-name>.log,
# appends one status line to <log-dir>/summary.txt ("[ok]" / "[error]") and,
# on failure, copies the last TAIL lines of the job's log into summary.txt.
# Exits with the command's exit status so make can mark the target failed.
set -u

name=$1
shift
logdir=$1
shift

mkdir -p "$logdir"
log="$logdir/$name.log"

# Run the job in the background so an interrupt (Ctrl-C / SIGTERM from the
# parent make tree) can be forwarded to it, otherwise Rscript would survive as
# an orphan. Without this, killing the wrapper leaves the real job running.
#
# `set -m` gives the background job its own process group, so killing "-$child"
# (negative pid = process group) also takes out grandchildren the wrapper knows
# nothing about -- e.g. the PSOCK workers R's makeCluster() spawns. Killing the
# Rscript pid alone would leave those workers orphaned forever, since R's
# on.exit(stopCluster(...)) never runs when R dies from a signal.
set -m
"$@" >"$log" 2>&1 &
child=$!

kill_child() {
    kill -TERM -- -"$child" 2>/dev/null
    sleep "${KILL_GRACE:-5}"
    # Escalate in case anything ignored or survived the TERM.
    kill -KILL -- -"$child" 2>/dev/null
    kill -KILL -- "$child" 2>/dev/null
}
trap 'kill_child; exit 130' INT
trap 'kill_child; exit 143' TERM

if wait "$child"; then
    printf '[ok]    %s\n' "$name" | tee -a "$logdir/summary.txt"
    exit 0
fi

rc=$?
{
    printf '[error] %s\n' "$name"
    printf '\n--- tail of %s ---\n' "$log"
    tail -n "${TAIL:-30}" "$log"
    printf '\n'
} >>"$logdir/summary.txt"
printf '[error] %s  (%s)\n' "$name" "$log" >&2
exit $rc
