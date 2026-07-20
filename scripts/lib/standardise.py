"""Generic model/benchmark standardisation operations.

This is the reusable core that the seven historical one-off
"standardise pass" scripts (fix_setup_in_names.py, standardise_models.py,
standardise_pass3.py-pass7.py) plus deduplicate_models.py and
merge_duplicate_benchmarks.py each reimplemented with a hardcoded map.
Every function here takes its map as a parameter, so the same code
applies any past or future pass -- build a JSON rules file and feed it to
`scripts/standardise.py`.

The five model/benchmark operations, all faithful to the proven pass
logic:

  REMOVE         cascade-delete a model row + all its results rows
                 (a model that should never have been imported).
  RENAME         rename a model_id in place across models.csv +
                 results.csv (same model, canonical id).
  REMAP          merge a model into an existing canonical model_id:
                 repoint its results, delete its now-redundant models row
                 (the canonical row + its metadata are left untouched).
  SETUP_EXTRACT  a model whose name encodes an eval technique (e.g.
                 "...-cot", "...-sc") is folded into its base model, with
                 the technique moved into results.setup (only where setup
                 is empty, so an explicit setup is never clobbered).
  MERGE_BENCHMARK relabel a duplicate benchmark_id to its canonical id,
                 porting any non-empty metadata into empty canonical
                 fields, then drop the duplicate benchmark row.

Result rows that become identical after a merge are collapsed with
`dedup_results` (drop_duplicates on config.RESULT_IDENTITY_KEY), matching
what every pass did as its final step; genuine multi-setup / multi-source
rows differ on the key and are preserved.
"""
from . import config, integrity


# ── model operations ──────────────────────────────────────────────────────

def apply_remove(models, results, remove_ids):
    """Cascade-delete each model_id in `remove_ids` from models.csv and
    every matching results.csv row. `remove_ids` may be a list, or a
    {id: reason} dict (the reason is only used for the log line).
    Returns (models, results, report)."""
    reasons = remove_ids if isinstance(remove_ids, dict) else {i: "" for i in remove_ids}
    report = {"removed_models": 0, "removed_results": 0, "missing": []}
    present = set(models["model_id"])

    for old_id, reason in reasons.items():
        if old_id not in present:
            report["missing"].append(old_id)
            print(f"  WARN  REMOVE source missing: {old_id!r}")
            continue
        n = int((results["model_name"] == old_id).sum())
        tag = f"  [{reason}]" if reason else ""
        print(f"  REMOVE  {old_id!r}  ({n}r){tag}")
        models = models[models["model_id"] != old_id]
        results = results[results["model_name"] != old_id]
        report["removed_models"] += 1
        report["removed_results"] += n

    return models.reset_index(drop=True), results.reset_index(drop=True), report


def apply_rename(models, results, rename_map):
    """Rename {old_id: new_id} in place: same model, canonical id. Updates
    models.model_id/model_name and results.model_name/model_id.
    Returns (models, results, report)."""
    report = {"renamed_models": 0, "renamed_results": 0, "missing": []}
    models = models.copy()
    results = results.copy()
    present = set(models["model_id"])

    for old_id, new_id in rename_map.items():
        if old_id not in present:
            report["missing"].append(old_id)
            print(f"  WARN  RENAME source missing: {old_id!r}")
            continue
        n = int((results["model_name"] == old_id).sum())
        print(f"  RENAME  {old_id!r} -> {new_id!r}  ({n}r)")
        models.loc[models["model_id"] == old_id, "model_id"] = new_id
        models.loc[models["model_name"] == old_id, "model_name"] = new_id
        results.loc[results["model_name"] == old_id, "model_name"] = new_id
        if "model_id" in results.columns:
            results.loc[results["model_id"] == old_id, "model_id"] = new_id
        report["renamed_models"] += 1
        report["renamed_results"] += n

    return models, results, report


def apply_remap(models, results, remap_map):
    """Merge {old_id: canonical_id} into existing canonical models:
    repoint results to the canonical id and delete the redundant old
    models.csv row (the canonical row + metadata are left untouched).
    Returns (models, results, report)."""
    report = {"remapped": 0, "remapped_results": 0, "missing": [], "missing_target": []}
    models = models.copy()
    results = results.copy()
    present = set(models["model_id"])

    for old_id, canon_id in remap_map.items():
        if old_id not in present:
            report["missing"].append(old_id)
            print(f"  WARN  REMAP source missing: {old_id!r}")
            continue
        if canon_id not in present:
            report["missing_target"].append(canon_id)
            print(f"  WARN  REMAP target missing (will repoint anyway): {canon_id!r}")
        n = int((results["model_name"] == old_id).sum())
        print(f"  REMAP  {old_id!r} -> {canon_id!r}  ({n}r)")
        results.loc[results["model_name"] == old_id, "model_name"] = canon_id
        if "model_id" in results.columns:
            results.loc[results["model_id"] == old_id, "model_id"] = canon_id
        models = models[models["model_id"] != old_id]
        report["remapped"] += 1
        report["remapped_results"] += n

    return models.reset_index(drop=True), results, report


def apply_setup_extract(models, results, setup_map):
    """Fold a technique-qualified model into its base model, moving the
    technique into results.setup. `setup_map` maps
    {old_id: {"base": base_id, "setup": setup_val}}. The setup value is
    written only where results.setup is currently empty, so an explicit
    setup is never overwritten; the old models.csv row is deleted.
    Returns (models, results, report)."""
    report = {"extracted": 0, "extracted_results": 0, "missing": []}
    models = models.copy()
    results = results.copy()
    present = set(models["model_id"])

    for old_id, spec in setup_map.items():
        base_id, setup_val = spec["base"], spec["setup"]
        if old_id not in present:
            report["missing"].append(old_id)
            print(f"  WARN  SETUP_EXTRACT source missing: {old_id!r}")
            continue
        n = int((results["model_name"] == old_id).sum())
        print(f"  SETUP_EXTRACT  {old_id!r} -> {base_id!r}  setup={setup_val!r}  ({n}r)")
        mask = results["model_name"] == old_id
        results.loc[mask & (results["setup"] == ""), "setup"] = setup_val
        results.loc[mask, "model_name"] = base_id
        if "model_id" in results.columns:
            results.loc[results["model_id"] == old_id, "model_id"] = base_id
        models = models[models["model_id"] != old_id]
        report["extracted"] += 1
        report["extracted_results"] += n

    return models.reset_index(drop=True), results, report


# ── benchmark operations ────────────────────────────────────────────────────

def merge_benchmarks(benchmarks, results, merge_map):
    """Relabel {merged_id: canonical_id} duplicate/variant benchmarks.
    Each value may be a plain string canonical_id (metadata ported, no
    language backfill -- the historical behavior) or a dict
    {"canonical": canonical_id, "language": value} -- for consolidating a
    per-language benchmark variant into a shared id while preserving which
    language each row came from: every reassigned result row gets
    results.language backfilled to `value`, ONLY where currently blank
    (never overwrites an already-populated language, e.g. an existing
    "avg N langs" row already on the canonical benchmark).

    Ports any non-empty metadata from the merged row into *empty* fields
    of the canonical row, reassigns results.benchmark_id, then drops the
    merged benchmark row. Returns (benchmarks, results, report)."""
    report = {"merged": 0, "reassigned_results": 0, "backfilled_language": 0,
              "missing": [], "missing_target": []}
    b = benchmarks.copy()
    r = results.copy()

    for merged_id, spec in merge_map.items():
        if isinstance(spec, dict):
            canon_id, language = spec["canonical"], spec.get("language")
        else:
            canon_id, language = spec, None

        if merged_id not in set(b["benchmark_id"]):
            report["missing"].append(merged_id)
            print(f"  WARN  MERGE source missing: {merged_id!r}")
            continue
        if canon_id not in set(b["benchmark_id"]):
            report["missing_target"].append(canon_id)
            print(f"  ERROR MERGE target missing: {canon_id!r}")
            continue
        n = int((r["benchmark_id"] == merged_id).sum())
        tag = f"  language={language!r}" if language else ""
        print(f"  MERGE_BENCHMARK  {merged_id!r} -> {canon_id!r}  ({n}r){tag}")

        mrow = b[b["benchmark_id"] == merged_id].iloc[0]
        cmask = b["benchmark_id"] == canon_id
        ported = []
        for col in b.columns:
            if col == "benchmark_id":
                continue
            cval = b.loc[cmask, col].values[0]
            mval = mrow[col]
            if (cval is None or str(cval).strip() == "") and str(mval).strip() != "":
                b.loc[cmask, col] = mval
                ported.append(col)
        if ported:
            print(f"           ported metadata into canonical: {', '.join(ported)}")

        mmask = r["benchmark_id"] == merged_id
        if language and "language" in r.columns:
            blank_lang = mmask & (r["language"].str.strip() == "")
            report["backfilled_language"] += int(blank_lang.sum())
            r.loc[blank_lang, "language"] = language
        r.loc[mmask, "benchmark_id"] = canon_id
        b = b[b["benchmark_id"] != merged_id]
        report["merged"] += 1
        report["reassigned_results"] += n

    return b.reset_index(drop=True), r, report


def apply_remove_benchmark(benchmarks, results, remove_map):
    """Cascade-delete each benchmark_id in `remove_map` from
    benchmarks.csv and every matching results.csv row -- for a benchmark
    variant that should be dropped entirely rather than relabeled/kept
    under any id (e.g. a machine/human-translated per-language leaderboard
    split whose question set is a literal translation of an
    original-language benchmark already present, or a zero-result stub
    left behind by a model removal). `remove_map` may be a list, or a
    {benchmark_id: reason} dict (reason used only for the log line).
    Returns (benchmarks, results, report)."""
    reasons = remove_map if isinstance(remove_map, dict) else {i: "" for i in remove_map}
    report = {"removed_benchmarks": 0, "removed_results": 0, "missing": []}
    present = set(benchmarks["benchmark_id"])

    for bid, reason in reasons.items():
        if bid not in present:
            report["missing"].append(bid)
            print(f"  WARN  REMOVE_BENCHMARK source missing: {bid!r}")
            continue
        n = int((results["benchmark_id"] == bid).sum())
        tag = f"  [{reason}]" if reason else ""
        print(f"  REMOVE_BENCHMARK  {bid!r}  ({n}r){tag}")
        benchmarks = benchmarks[benchmarks["benchmark_id"] != bid]
        results = results[results["benchmark_id"] != bid]
        report["removed_benchmarks"] += 1
        report["removed_results"] += n

    return benchmarks.reset_index(drop=True), results.reset_index(drop=True), report


def cascade_remove_benchmarks(benchmarks, models, results, remove_map):
    """apply_remove_benchmark(), then cascade onto the model axis: any
    model left with zero result rows once those benchmarks are gone is
    itself removed via apply_remove(). One pass in each direction is
    provably sufficient -- an orphaned model has no result rows left to
    delete, so removing it cannot itself orphan any (other) benchmark.
    Returns (benchmarks, models, results, report)."""
    benchmarks, results, bench_report = apply_remove_benchmark(benchmarks, results, remove_map)

    orphans = integrity.check_orphans(benchmarks, models, results)["orphan_models"]
    orphan_reasons = {m: "Orphaned by benchmark cascade removal" for m in orphans}
    models, results, model_report = apply_remove(models, results, orphan_reasons)

    report = {
        "removed_benchmarks": bench_report["removed_benchmarks"],
        "removed_results": bench_report["removed_results"],
        "missing_benchmarks": bench_report["missing"],
        "orphaned_models": model_report["removed_models"],
        "orphan_model_ids": sorted(orphans),
    }
    return benchmarks, models, results, report


# ── cleanups ────────────────────────────────────────────────────────────────

def dedup_results(results):
    """Collapse result rows that became identical after a merge (exact
    duplicates on config.RESULT_IDENTITY_KEY). Returns (results, dropped)."""
    key_cols = [c for c in config.RESULT_IDENTITY_KEY if c in results.columns]
    before = len(results)
    deduped = results.drop_duplicates(subset=key_cols).reset_index(drop=True)
    return deduped, before - len(deduped)


def dedup_models(models):
    """Drop duplicate models.csv rows on model_id (keep first). Returns
    (models, dropped)."""
    before = len(models)
    deduped = models.drop_duplicates(subset="model_id", keep="first").reset_index(drop=True)
    return deduped, before - len(deduped)
