#!/usr/bin/env python3
"""
Family Audit — Standardisation Pass 4
======================================
Fixes remaining issues from the model-family audit:

  1. GPT-2 HF-style IDs → canonical
  2. Mistral/Mixtral 7B and 8x7B consolidation
  3. Setup-in-name extraction (maj@k, self-consistency, w/ code)
     including LLaMA maj@k, Minerva maj@k, OpenMath (w/ code) variants
  4. PaLM (Self Consistency) → setup column

RENAME  — update model_id in-place (no existing canonical target)
REMAP   — merge into existing canonical, delete old entry
SETUP_REMAP — merge into base + move qualifier to results.setup

Usage:
  python3 scripts/standardise_pass4.py           # dry run
  python3 scripts/standardise_pass4.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io, config

# ─────────────────────────────────────────────────────────────────────────────
# RENAME — no existing canonical; update model entry in-place.
# Applied FIRST so REMAP/SETUP_REMAP targets resolve correctly.
# ─────────────────────────────────────────────────────────────────────────────
RENAME = {
    # GPT-2 HF-style IDs → canonical with readable size suffix
    "gpt2-large":  "GPT-2 Large (774M)",   # 14r
    "gpt2-medium": "GPT-2 Medium (355M)",  # 8r

    # OpenMath base names (created so SETUP_REMAP targets exist)
    "OpenMath-CodeLlama-70B (w/ code)":  "OpenMath-CodeLlama-70B",   # 5r → base
    "OpenMath-CodeLlama-34B (w/ code)":  "OpenMath-CodeLlama-34B",   # 2r → base
    "OpenMath-CodeLlama-13B (w/ code)":  "OpenMath-CodeLlama-13B",   # 2r → base
    "OpenMath-CodeLlama-7B (w/ code)":   "OpenMath-CodeLlama-7B",    # 2r → base
    "OpenMath-Llama2-70B (w/ code)":     "OpenMath-Llama2-70B",      # 2r → base
    "OpenMath-Mistral-7B (w/ code)":     "OpenMath-Mistral-7B",      # 2r → base
}

# ─────────────────────────────────────────────────────────────────────────────
# REMAP — merge into existing canonical; delete old entry.
# ─────────────────────────────────────────────────────────────────────────────
REMAP = {
    # ── GPT-2 ─────────────────────────────────────────────────────────────────
    "gpt2-xl":    "GPT-2-XL 1.5B",  # 6r  (after RENAME, REMAP to existing 5r entry)

    # ── Mistral 7B ────────────────────────────────────────────────────────────
    # "Mistral 7B" PwC scores (MBPP 47.5, NQ 28.8, ARC 55.5/80.5, PIQA 83.0, etc.)
    # exactly match the published Mistral 7B v0.1 numbers.
    "Mistral 7B":       "Mistral v0.1 (7B)",  # 7r
    # HF leaderboard model ID, explicitly v0.1
    "mistral-7b-v0.1":  "Mistral v0.1 (7B)",  # 15r

    # ── Mixtral 8x7B base ─────────────────────────────────────────────────────
    # HELM names this "Mixtral (8x7B 32K seqlen)" (32K = context window, not a separate model)
    "Mixtral (8x7B 32K seqlen)":  "Mixtral-8x7B-v0.1",  # 10r
    # PwC and paper-sourced IDs for the same base model
    "Mixtral 8x7B":   "Mixtral-8x7B-v0.1",  # 4r
    "Mixtral-8x7B":   "Mixtral-8x7B-v0.1",  # 11r

    # ── Mixtral 8x7B instruct ─────────────────────────────────────────────────
    "Mixtral-8x7B-Instruct":       "Mixtral Instruct (8x7B)",  # 5r
    "mixtral-8x7b-instruct-v0.1":  "Mixtral Instruct (8x7B)",  # 17r
}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP_REMAP — old_id → (base_id, setup_val)
# Moves the setup qualifier to results.setup, then merges into base model.
# Applied AFTER RENAME so base names exist.
# ─────────────────────────────────────────────────────────────────────────────
SETUP_REMAP = {
    # ── PaLM self-consistency ─────────────────────────────────────────────────
    "PaLM 540B (Self Consistency)":   ("PaLM 540B",  "Self-Consistency"),  # 6r

    # ── LLaMA maj@k ───────────────────────────────────────────────────────────
    "LLaMA 65B (maj1@k)":   ("LLaMA 65B",  "maj@k"),   # 2r
    "LLaMA 7B (maj1@k)":    ("LLaMA 7B",   "maj@k"),   # 2r
    "LLaMA 13B-maj1@k":     ("LLaMA 13B",  "maj@k"),   # 2r
    "LLaMA 33B-maj1@k":     ("LLaMA 33B",  "maj@k"),   # 2r

    # ── Minerva maj@k ─────────────────────────────────────────────────────────
    "Minerva 540B (maj1@k, k=64)":  ("Minerva 540B", "maj@64"),  # 1r
    "Minerva 62B (maj1@k, k=64)":   ("Minerva 62B",  "maj@64"),  # 1r
    "Minerva 8B (maj1@k, k=64)":    ("Minerva 8B",   "maj@64"),  # 1r

    # ── OpenMath SC variants (base created by RENAME above) ───────────────────
    "OpenMath-CodeLlama-70B (w/ code, SC, k=50)": ("OpenMath-CodeLlama-70B", "SC k=50"),  # 2r
    "OpenMath-CodeLlama-34B (w/ code, SC, k=50)": ("OpenMath-CodeLlama-34B", "SC k=50"),  # 2r
    "OpenMath-CodeLlama-13B (w/ code, SC, k=50)": ("OpenMath-CodeLlama-13B", "SC k=50"),  # 2r
    "OpenMath-CodeLlama-7B (w/ code, SC, k=50)":  ("OpenMath-CodeLlama-7B",  "SC k=50"),  # 2r
    "OpenMath-Llama2-70B (w/ code, SC, k=50)":    ("OpenMath-Llama2-70B",    "SC k=50"),  # 2r
    "OpenMath-Mistral-7B (w/ code, SC, k=50)":    ("OpenMath-Mistral-7B",    "SC k=50"),  # 2r

    # ── Additional mCoT/CoT/SC setups (applied inline after pass4 --write) ────
    "GAL 120B (5-shot) mCoT":     ("GAL 120B",     "5-shot mCoT"),   # 1r
    "GAL 30B (5-shot) mCoT":      ("GAL 30B",      "5-shot mCoT"),   # 1r
    "PaLM 540B (5-shot) mCoT":    ("PaLM 540B",    "5-shot mCoT"),   # 1r
    "Minerva 540B (5-shot) mCoT": ("Minerva 540B", "5-shot mCoT"),   # 1r
    "Codex 5-shot CoT":           ("Codex",         "5-shot CoT"),   # 2r
    "ToRA-70B (SC, k=50)":        ("ToRA 70B",      "SC k=50"),      # 1r
    "ToRA-Code-34B (SC, k=50)":   ("ToRA-Code 34B", "SC k=50"),      # 1r
}
# Note: "w/ code" qualifier is moved to setup for the RENAME'd base entries too;
# we patch it at write time for the RENAME targets.
# OpenMath RENAMEs above set the base name; we also update their setup column.
RENAME_SETUP = {
    # old_id: setup_val to set on result rows AFTER RENAME (base already renamed)
    "OpenMath-CodeLlama-70B": "w/ code",   # was (w/ code), setup not set
    "OpenMath-CodeLlama-34B": "w/ code",
    "OpenMath-CodeLlama-13B": "w/ code",
    "OpenMath-CodeLlama-7B":  "w/ code",
    "OpenMath-Llama2-70B":    "w/ code",
    "OpenMath-Mistral-7B":    "w/ code",
}


def validate(m_df):
    existing = set(m_df["model_id"])
    will_exist = existing | set(RENAME.values())
    ok = True
    for old_id, new_id in RENAME.items():
        if old_id not in existing:
            print(f"  WARN  RENAME source missing: {old_id!r}")
        if new_id in existing:
            print(f"  ERROR RENAME target already exists (use REMAP): {new_id!r}")
            ok = False
    for old_id, new_id in REMAP.items():
        if old_id not in existing:
            print(f"  WARN  REMAP source missing: {old_id!r}")
        if new_id not in will_exist:
            print(f"  ERROR REMAP target missing: {new_id!r}")
            ok = False
    for old_id, (base_id, _) in SETUP_REMAP.items():
        if old_id not in existing:
            print(f"  WARN  SETUP_REMAP source missing: {old_id!r}")
        if base_id not in will_exist:
            print(f"  ERROR SETUP_REMAP base missing: {base_id!r}")
            ok = False
    return ok


def apply(m_df, r_df, write: bool):
    m = m_df.copy()
    r = r_df.copy()

    # ── 1. RENAME ─────────────────────────────────────────────────────────────
    for old_id, new_id in RENAME.items():
        cur_ids = set(m["model_id"]) if write else set(m_df["model_id"])
        if old_id not in cur_ids:
            print(f"  WARN  RENAME source missing: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  RENAME  {old_id!r} -> {new_id!r}  ({n}r)")
        if write:
            m.loc[m["model_id"]   == old_id, "model_id"]   = new_id
            m.loc[m["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_id"]   == old_id, "model_id"]   = new_id

    # Set setup for OpenMath RENAMEs (they had "w/ code" in old name, now in setup)
    if write:
        for new_id, setup_val in RENAME_SETUP.items():
            mask = (r["model_name"] == new_id) & (r["setup"] == "")
            r.loc[mask, "setup"] = setup_val

    # ── 2. REMAP ──────────────────────────────────────────────────────────────
    for old_id, new_id in REMAP.items():
        cur_ids = set(m["model_id"]) if write else set(m_df["model_id"])
        if old_id not in cur_ids:
            print(f"  WARN  REMAP source missing: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  REMAP  {old_id!r} -> {new_id!r}  ({n}r)")
        if write:
            r.loc[r["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_id"]   == old_id, "model_id"]   = new_id
            m = m[m["model_id"] != old_id]

    # ── 3. SETUP_REMAP ────────────────────────────────────────────────────────
    for old_id, (base_id, setup_val) in SETUP_REMAP.items():
        cur_ids = set(m["model_id"]) if write else set(m_df["model_id"])
        if old_id not in cur_ids:
            print(f"  WARN  SETUP_REMAP source missing: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  SETUP_REMAP  {old_id!r}")
        print(f"               -> base={base_id!r}  setup={setup_val!r}  ({n}r)")
        if write:
            mask = r["model_name"] == old_id
            r.loc[mask & (r["setup"] == ""), "setup"] = setup_val
            r.loc[mask, "model_name"] = base_id
            r.loc[r["model_id"] == old_id, "model_id"] = base_id
            m = m[m["model_id"] != old_id]

    # ── 4. Post-merge dedup ───────────────────────────────────────────────────
    dupes_dropped = 0
    if write:
        key_cols = [c for c in config.RESULT_IDENTITY_KEY if c in r.columns]
        before = len(r)
        r = r.drop_duplicates(subset=key_cols).reset_index(drop=True)
        dupes_dropped = before - len(r)

    return m, r, dupes_dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    _, m_df, r_df = io.load_data()
    print(f"  models: {len(m_df)}  results: {len(r_df)}")

    print("\nValidating maps...")
    if not validate(m_df):
        print("Validation failed — fix errors above before applying.")
        sys.exit(1)
    print("  All targets verified ✓")

    print(f"\n{'Applying' if args.write else 'Dry run —'} changes:\n")
    m_new, r_new, n_dupes = apply(m_df, r_df, write=args.write)

    if args.write:
        io.save_csv(m_new, config.MODELS_CSV)
        io.save_results(r_new)
        print(f"\nWritten.")
        print(f"  models:  {len(m_df)} → {len(m_new)}  (−{len(m_df)-len(m_new)})")
        print(f"  results: {len(r_df)} → {len(r_new)}  "
              f"({'−' if len(r_new) <= len(r_df) else '+'}{abs(len(r_df)-len(r_new))}, "
              f"{n_dupes} post-merge dupes dropped)")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        n_rename = len(RENAME)
        n_remap = len(REMAP)
        n_setup = len(SETUP_REMAP)
        print(f"\nDry run summary:")
        print(f"  RENAME: {n_rename}  REMAP: {n_remap}  SETUP_REMAP: {n_setup}")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
