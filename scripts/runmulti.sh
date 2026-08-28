#!/usr/bin/env bash
# Fan a method list out into one parallel process per method.
#
# Usage: runmulti.sh <logdir> <kind> [suffix-tokens...] -- <method...> -- <roots...>
#
#   kind      impute | factor            -> src/run/<kind>.R
#   suffix    flags applied to every method (e.g. "raw", "loco"), used to name
#             the job and turned into --<token> flags for the R script
#   method    one per desired process; each gets its own Rscript invocation
#   roots     --data-root ... --results-root ..., forwarded to every process
#
# Example:  runmulti.sh logs factor raw loco -- knn missforest -- <roots>
# runs, in parallel:
#   Rscript src/run/factor.R --method knn --raw --loco <roots>
#   Rscript src/run/factor.R --method missforest --raw --loco <roots>
set -u
JOBS=${JOBS:-8}

logdir=$1; shift
kind=$1; shift

suffix=()
while [[ $# -gt 0 && $1 != "--" ]]; do suffix+=("$1"); shift; done
shift  # the first --
methods=()
while [[ $# -gt 0 && $1 != "--" ]]; do methods+=("$1"); shift; done
shift  # the second --
roots=("$@")

mkdir -p "$logdir"

namebase=$kind
for s in "${suffix[@]}"; do namebase="${namebase}-${s}"; done

flags=()
for s in "${suffix[@]}"; do flags+=("--${s}"); done
[[ $kind == impute ]] && flags+=(--reimpute)

if ((${#methods[@]} == 0)); then
    printf '[error] runmulti: no methods given for %s\n' "$kind" >&2
    exit 2
fi

pids=()
running=0

# Traps must be armed BEFORE any job is launched, otherwise an interrupt that
# arrives during the spawn loop is unhandled and leaves every already-started
# job running as an orphan.
kill_children() {
    # Each pid is a runone.sh wrapper; its own trap forwards the TERM to the
    # job's whole process group (Rscript + any workers it spawned). Escalate
    # to KILL in case a wrapper is stuck or already gone.
    kill -TERM "${pids[@]}" 2>/dev/null
    sleep "${KILL_GRACE:-5}"
    kill -KILL "${pids[@]}" 2>/dev/null
}
trap 'kill_children; exit 130' INT
trap 'kill_children; exit 143' TERM

for m in "${methods[@]}"; do
    ./scripts/runone.sh "${namebase}-${m}" "$logdir" \
        Rscript "src/run/${kind}.R" --method "$m" "${flags[@]}" "${roots[@]}" &
    pids+=($!)
    if (( ++running >= JOBS )); then
        wait -n
        ((--running))
    fi
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
