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
"$@" >"$log" 2>&1 &
child=$!

kill_child() { kill -TERM -- "$child" 2>/dev/null; sleep 0.2; kill -INT -- "$child" 2>/dev/null; }
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
