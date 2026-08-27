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
for m in "${methods[@]}"; do
    ./scripts/runone.sh "${namebase}-${m}" "$logdir" \
        Rscript "src/run/${kind}.R" --method "$m" "${flags[@]}" "${roots[@]}" &
    pids+=($!)
    if (( ++running >= JOBS )); then
        wait -n
        ((--running))
    fi
done

# Forward SIGINT/SIGTERM to every worker (each forwards it on to its Rscript),
# so an interrupt doesn't leave orphaned R processes running.
kill_children() { kill -TERM "${pids[@]}" 2>/dev/null; }
trap 'kill_children; exit 130' INT
trap 'kill_children; exit 143' TERM

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
