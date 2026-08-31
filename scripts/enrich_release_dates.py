#!/usr/bin/env python3
"""Fill `release_date` (YYYY-MM floor) on benchmarks.csv and models.csv.

Stage one only: everything derivable from data already in the tables, at zero
lookup cost (arXiv identifiers for benchmarks, embedded date stamps for models).
Whatever this leaves unfilled is the work list for stage two -- external
per-row research, delegated to the hermes agent -- and `--todo` writes exactly
that list out so stage two never re-researches an answer we already hold.

Writes to the ARCHIVE (`data/*.csv`), because a release date is a fact about
the model or benchmark rather than a modelling decision; it reaches the
analysis view on the next `make_text_only_copy.py` run.

    python3 scripts/enrich_release_dates.py            # dry run (default)
    python3 scripts/enrich_release_dates.py --write    # persist
    python3 scripts/enrich_release_dates.py --todo DIR # emit stage-two work lists
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib import config, io, release_dates


def _summarise(label, report, derived_key):
    filled = report["existing"] + report[derived_key]
    print(f"\n{label} ({report['total']} rows)")
    print(f"  kept existing valid : {report['existing']}")
    print(f"  derived ({derived_key:10s}): {report[derived_key]}")
    print(f"  -> month precision  : {filled} ({100 * filled / report['total']:.0f}%)")
    print(f"  still missing       : {report['missing']}  <- stage two (hermes)")
    if report["cleared_invalid"]:
        print(f"  cleared {len(report['cleared_invalid'])} invalid value(s) "
              f"(not a YYYY-MM date):")
        for key, bad in report["cleared_invalid"]:
            print(f"    {key}: {bad!r}")


def _write_todo(out_dir, benchmarks, models):
    out_dir.mkdir(parents=True, exist_ok=True)
    b = benchmarks[benchmarks["release_date"].isna()].copy()
    # Carry whatever seeds we have: a year narrows the search, and the name and
    # URLs are what an agent needs to identify the thing at all.
    bcols = [c for c in ["benchmark_id", "benchmark_name", "year", "venue",
                         "paper_url", "source_url", "organization"]
             if c in b.columns]
    bp = out_dir / "todo_benchmarks.csv"
    b[bcols].to_csv(bp, index=False)

    m = models[models["release_date"].isna()].copy()
    mcols = [c for c in ["model_id", "model_name", "model_family", "developer",
                         "organization", "model_type", "url", "year_evaluated"]
             if c in m.columns]
    mp = out_dir / "todo_models.csv"
    m[mcols].to_csv(mp, index=False)

    # io.load_data() keeps missing values as empty strings, so notna() alone
    # counts blanks as populated -- require a non-empty value.
    def _filled(df, col):
        if col not in df.columns:
            return 0
        return int(df[col].fillna("").astype(str).str.strip().ne("").sum())

    print(f"\nStage-two work lists:")
    print(f"  {bp}  ({len(b)} benchmarks; {_filled(b, 'year')} have a year "
          f"already, so only the month is unknown)")
    print(f"  {mp}  ({len(m)} models; {_filled(m, 'developer')} have a "
          f"developer, {_filled(m, 'model_family')} a family)")


def _merge_pass(dirpath, kind, key_col, frame, label_a, label_b):
    """Apply the agreement filter to one entity kind. Returns (frame, stats)."""
    import pandas as pd

    a_path = dirpath / f"pass_a_{kind}.csv"
    b_path = dirpath / f"pass_b_{kind}.csv"
    if not (a_path.exists() and b_path.exists()):
        print(f"  {kind}: skipped — need both {a_path.name} and {b_path.name}")
        return frame, None

    a = release_dates.read_answer_csv(a_path)
    b = release_dates.read_answer_csv(b_path)

    accepted, rejected = {}, []
    for k in set(a) | set(b):
        value, source, reason = release_dates.reconcile_with_fallback(
            a.get(k), b.get(k), label_a, label_b)
        if value:
            accepted[k] = (value, source)
        else:
            rejected.append((k, a.get(k), b.get(k), reason))

    # Only ever fill blanks. A stage-one value is derived from the data itself
    # and outranks anything researched, so it is never overwritten here.
    filled = 0
    dates = frame["release_date"].tolist()
    sources = frame["release_date_source"].tolist()
    for i, k in enumerate(frame[key_col].tolist()):
        cur = dates[i]
        if cur is not None and str(cur).strip() not in ("", "nan"):
            continue
        if k in accepted:
            dates[i], sources[i] = accepted[k]
            filled += 1
    frame["release_date"] = dates
    frame["release_date_source"] = sources

    rej_path = dirpath / f"rejected_{kind}.csv"
    pd.DataFrame(rejected, columns=["key", "pass_a", "pass_b", "reason"]).to_csv(
        rej_path, index=False)

    from collections import Counter
    tally = Counter(s for _, s in accepted.values())
    corr = tally["corroborated_month"] + tally["corroborated_year"]
    singles = {k: v for k, v in tally.items() if k.startswith("single_")}
    print(f"  {kind}: {len(accepted)} accepted "
          f"[corroborated {corr} ({tally['corroborated_month']} month, "
          f"{tally['corroborated_year']} year); "
          + ", ".join(f"{k} {v}" for k, v in sorted(singles.items())) + "]")
    print(f"    {len(rejected)} rejected -> {rej_path.name}; filled {filled} blank rows")
    return frame, {"accepted": len(accepted), "rejected": len(rejected)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="persist changes")
    ap.add_argument("--todo", metavar="DIR", default=None,
                    help="write stage-two work lists to DIR")
    ap.add_argument("--merge", metavar="DIR", default=None,
                    help="merge stage-two pass_a_*/pass_b_* results from DIR "
                         "through the two-pass agreement filter")
    args = ap.parse_args(argv)

    benchmarks, models, results = io.load_data()
    benchmarks, brep = release_dates.enrich_benchmarks(benchmarks)
    models, mrep = release_dates.enrich_models(models)

    _summarise("BENCHMARKS", brep, "arxiv_id")
    _summarise("MODELS", mrep, "name_stamp")

    if args.todo:
        _write_todo(Path(args.todo), benchmarks, models)

    if args.merge:
        d = Path(args.merge)
        print(f"\nTwo-pass agreement merge from {d}:")
        # benchmarks: both passes were hermes. models: pass B is haiku.
        benchmarks, _ = _merge_pass(d, "benchmarks", "benchmark_id", benchmarks,
                                    "hermes", "hermes")
        models, _ = _merge_pass(d, "models", "model_id", models,
                                "hermes", "haiku")

    if args.write:
        io.save_csv(benchmarks, config.BENCHMARKS_CSV)
        io.save_csv(models, config.MODELS_CSV)
        print(f"\nWrote {config.BENCHMARKS_CSV} and {config.MODELS_CSV}")
        print("Next: python3 scripts/verify_data.py && "
              "python3 scripts/make_text_only_copy.py")
    else:
        print("\nDry run only — pass --write to persist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
