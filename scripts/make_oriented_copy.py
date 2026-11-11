#!/usr/bin/env python3
"""Build the REVISION view: data/text_only/ + scale fixes + metric orientation.

Reads data/text_only/ (never writes to it) and writes a sibling root, by
default data/text_only_oriented/, that the whole pipeline can run on unchanged:

    uv run python scripts/make_oriented_copy.py                 # --orient documented
    make preproc  DATA_ROOT=data/text_only_oriented DENSITY=0.1
    make runall   DATA_ROOT=data/text_only_oriented RESULTS_ROOT=results/text_only_oriented DENSITY=0.1

Three passes, each driven by a knowledge base in scripts/lib/config.py:

  1. SCALE_CORRECTIONS -- rows of a column stored on the wrong scale by the
     extraction-time x100 rule (the_pile: PwC "Bits per byte" stored as 55-96).
  2. REVISION_SOURCE_SCALE_CONFLICTS -- same mechanism as gpqa: keep the Open
     LLM Leaderboard v2 convention for bbh / math, drop the minority source.
  3. ORIENTATION -- negate every benchmark in LOWER_IS_BETTER_BENCHMARKS
     (and, with --orient all, AMBIGUOUS_DIRECTION_BENCHMARKS), so that larger
     is better in every column.

--orient none applies only passes 1-2: the "bug-fix only" arm, which isolates
the effect of orientation from the effect of the scale fixes in an A/B run.

Why a separate root rather than editing data/text_only/: the paper's numbers
were produced on data/text_only/, and the reviewers ask how much the results
CHANGE under orientation. Keeping both roots makes that a diff of two runs.

The original score is preserved in `score_raw`; `score_transform` records what
was done to each row ("", "rescaled/100", "negated", "rescaled/100;negated").
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib import config, io  # noqa: E402

SRC_DIR = config.DATA_DIR / "text_only"
DEFAULT_OUT = config.DATA_DIR / "text_only_oriented"


def _fmt(x):
    """Format a float back to the dataset's plain-string convention."""
    return f"{x:.6f}".rstrip("0").rstrip(".") if pd.notna(x) else ""


def apply_scale_corrections(results, log):
    score = pd.to_numeric(results["score"], errors="coerce")
    for bid, rule in config.SCALE_CORRECTIONS.items():
        hit = ((results["benchmark_id"] == bid)
               & (results["source_name"] == rule["source_name"])
               & (results["metric_name"] == rule["metric_name"])
               & (score > rule["min_score"]))
        n = int(hit.sum())
        if n == 0:
            log.append(f"- SCALE {bid}: rule matched 0 rows -- stale entry?")
            continue
        new = score[hit] / rule["divide_by"]
        results.loc[hit, "score"] = new.map(_fmt)
        results.loc[hit, "score_transform"] = f"rescaled/{rule['divide_by']:g}"
        score[hit] = new
        log.append(f"- SCALE {bid}: {n} rows ({rule['source_name']} / "
                   f"{rule['metric_name']}) divided by {rule['divide_by']:g}; "
                   f"now {new.min():.4f}-{new.max():.4f}")
    return results


def apply_source_conflicts(results, log):
    drop = pd.Series(False, index=results.index)
    for bid, (keep_src, why) in config.REVISION_SOURCE_SCALE_CONFLICTS.items():
        sub = results["benchmark_id"] == bid
        if not sub.any():
            log.append(f"- SOURCE {bid}: benchmark absent -- stale entry?")
            continue
        keeping = sub & (results["source_name"] == keep_src)
        if not keeping.any():
            log.append(f"- SOURCE {bid}: '{keep_src}' absent -- NOT applied")
            continue
        d = sub & ~keeping
        drop |= d
        log.append(f"- SOURCE {bid}: kept {int(keeping.sum())} rows from "
                   f"'{keep_src}', dropped {int(d.sum())} "
                   f"({', '.join(sorted(results.loc[d, 'source_name'].unique()))})")
    return results[~drop].reset_index(drop=True)


def apply_orientation(results, orient, log):
    flip = {}
    if orient in ("documented", "all"):
        flip.update({b: f"{k}: {m}" for b, (k, m) in
                     config.LOWER_IS_BETTER_BENCHMARKS.items()})
    if orient == "all":
        flip.update({b: "ambiguous" for b in config.AMBIGUOUS_DIRECTION_BENCHMARKS})
    present = set(results["benchmark_id"])
    score = pd.to_numeric(results["score"], errors="coerce")
    for bid in sorted(flip):
        hit = results["benchmark_id"] == bid
        if bid not in present:
            log.append(f"- ORIENT {bid}: not in this view (skipped)")
            continue
        results.loc[hit, "score"] = (-score[hit]).map(_fmt)
        prev = results.loc[hit, "score_transform"]
        results.loc[hit, "score_transform"] = prev.where(prev == "", prev + ";") + "negated"
        results.loc[hit, "metric_lower_is_better"] = "False"
        log.append(f"- ORIENT {bid}: negated {int(hit.sum())} rows ({flip[bid]})")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orient", choices=["none", "documented", "all"],
                    default="documented",
                    help="none = scale fixes only; documented = also negate "
                         "LOWER_IS_BETTER_BENCHMARKS; all = also negate "
                         "AMBIGUOUS_DIRECTION_BENCHMARKS")
    ap.add_argument("--src", type=Path, default=SRC_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output root (default {DEFAULT_OUT.relative_to(config.REPO_ROOT)}"
                         "[_<orient> when --orient is not 'documented'])")
    args = ap.parse_args()
    out = args.out or (DEFAULT_OUT if args.orient == "documented"
                       else DEFAULT_OUT.with_name(f"{DEFAULT_OUT.name}_{args.orient}"))
    out.mkdir(parents=True, exist_ok=True)

    results = io.load_csv(args.src / "results.csv")
    n0 = len(results)
    results["score_raw"] = results["score"]
    results["score_transform"] = ""
    log = [f"# Revision view (`--orient {args.orient}`)", "",
           f"Source: `{args.src.relative_to(config.REPO_ROOT)}` ({n0} result rows). "
           "Generated by `scripts/make_oriented_copy.py`; do not hand-edit.", ""]

    results = apply_scale_corrections(results, log)
    results = apply_source_conflicts(results, log)
    if args.orient != "none":
        results = apply_orientation(results, args.orient, log)

    # Every other file is carried over verbatim. Benchmarks / models are not
    # cascaded: the source-conflict pass drops rows of benchmarks that keep
    # hundreds of other rows, so no benchmark is orphaned; an orphaned model
    # (scored only by a dropped source) simply never enters the matrix.
    # Only the dataset tables -- not densify outputs that may sit beside them.
    for name in ("benchmarks.csv", "models.csv", "benchmark_labels.csv"):
        if (args.src / name).exists():
            shutil.copy2(args.src / name, out / name)
    io.save_csv(results, out / "results.csv")
    log += ["", f"Result rows: {n0} -> {len(results)}."]
    (out / "README.md").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
