#!/usr/bin/env python3
"""
Family Audit — Standardisation Pass 3
======================================
Third pass fixing issues identified by the model-family audit (GPT, Llama,
Qwen, Mistral, PaLM, Gemma, Phi families).

Operations:
  RENAME  — rename a model_id in-place (no existing canonical target)
  REMAP   — merge results into an existing canonical model, delete old entry

Usage:
  python3 scripts/standardise_pass3.py           # dry run
  python3 scripts/standardise_pass3.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io, config

# ─────────────────────────────────────────────────────────────────────────────
# RENAME — old_id has no existing canonical; update model entry in-place.
# Apply BEFORE REMAP so REMAP targets resolve correctly.
# ─────────────────────────────────────────────────────────────────────────────
RENAME = {
    # GPT-5 naming convention: lowercase "mini"/"nano", space not hyphen
    "GPT-5-mini":  "GPT-5 mini",     # 17r — hyphen→space is canonical
    "GPT-5 Nano":  "GPT-5 nano",     # 2r  — uppercase→lowercase for consistency

    # Llama 4 Scout: "S" abbreviation used by one paper; no canonical entry yet
    "Llama-4-S":   "Llama 4 Scout",  # 66r
}

# ─────────────────────────────────────────────────────────────────────────────
# REMAP — merge into existing canonical; delete old entry.
# ─────────────────────────────────────────────────────────────────────────────
REMAP = {
    # ── GPT date-versioned variants → canonical ───────────────────────────────
    "GPT-4.1 (2025-04-14)":          "GPT-4.1",       # 12r
    "GPT-4.1 mini (2025-04-14)":     "GPT-4.1 mini",  # 12r
    "GPT-4.1 nano (2025-04-14)":     "GPT-4.1 nano",  # 12r
    "GPT-4.5 (2025-02-27 preview)":  "GPT-4.5",       # 5r
    "GPT-5 (2025-08-07)":            "GPT-5",          # 42r
    "GPT-5 mini (2025-08-07)":       "GPT-5 mini",    # 45r  (after RENAME)
    "GPT-5 nano (2025-08-07)":       "GPT-5 nano",    # 5r   (after RENAME)
    "GPT-5.1 (2025-11-13)":          "GPT-5.1",       # 12r
    "GPT-5.4 (2026-03-05)":          "GPT-5.4",       # 7r
    "GPT-5.4 mini (2026-03-17)":     "GPT-5.4 Mini",  # 7r
    "GPT-5.4 nano (2026-03-17)":     "GPT-5.4 Nano",  # 7r

    # ── GPT alias cleanup ─────────────────────────────────────────────────────
    "chatgpt5":       "GPT-5",         # 67r — ChatGPT 5 alias
    "chatgpt5_mini":  "GPT-5 mini",    # 1r   (after RENAME)
    "GPT 3.5":        "GPT-3.5",       # 1r   — space→hyphen
    "GPT-3.5 Turbo":  "GPT-3.5-Turbo",# 3r   — space→hyphen in canonical
    "gpt2":           "GPT-2",         # 12r  — lowercase HF-style → canonical

    # ── PaLM cleanup ─────────────────────────────────────────────────────────
    "PALM":       "PaLM",      # 6r  — wrong capitalisation
    "PaLM-540B":  "PaLM 540B",# 25r — hyphen→space
    "PaLM-62B":   "PaLM 62B", # 3r  — hyphen→space

    # ── Llama cleanup ────────────────────────────────────────────────────────
    "Llama-4-M":          "Llama 4 Maverick",       # 66r — "M" = Maverick
    "LLAMA-2":            "Llama 2",                 # 1r  — wrong case
    "Llama 2 70B":        "Llama 2 (70B)",           # 1r  — parentheses are canonical
    "Llama-2-70b-chat":   "Llama-2-70B-Chat",        # 1r  — casing fix
    "Llama-3.3-70B-Instruct": "Llama 3.3 Instruct (70B)",  # 15r — HF→canonical

    # ── Qwen cleanup ─────────────────────────────────────────────────────────
    "qwen-chat-14b":  "Qwen-14B-Chat",  # 2r
    "qwen-chat-7b":   "Qwen-7B-Chat",   # 1r

    # ── Mistral/Mixtral cleanup ───────────────────────────────────────────────
    "Mixtral-8x7b inst":  "Mixtral Instruct (8x7B)",  # 3r — informal name

    # ── Gemma cleanup ────────────────────────────────────────────────────────
    "Gemma 4 31B Instruct":  "Gemma 4 31B",   # 7r — Gemma 4 only ships instruct
    "Gemma-3-27B":           "Gemma 3 27B",   # 8r — hyphen→space canonical

    # ── Phi cleanup ──────────────────────────────────────────────────────────
    "phi-1_5":  "phi-1.5",  # 12r — underscore→dot
}


def validate(m_df):
    existing = set(m_df["model_id"])
    will_exist = existing | set(RENAME.values())
    ok = True
    for old_id, new_id in RENAME.items():
        if old_id not in existing:
            print(f"  WARN  RENAME source not found: {old_id!r}")
        if new_id in existing:
            print(f"  ERROR RENAME target already exists (use REMAP): {new_id!r}")
            ok = False
    for old_id, new_id in REMAP.items():
        if old_id not in existing:
            print(f"  WARN  REMAP source not found (already gone?): {old_id!r}")
        if new_id not in will_exist:
            print(f"  ERROR REMAP target missing: {new_id!r}")
            ok = False
    return ok


def apply(m_df, r_df, write: bool):
    m = m_df.copy()
    r = r_df.copy()
    existing = set(m["model_id"])

    # ── 1. RENAME ─────────────────────────────────────────────────────────────
    rename_count = 0
    for old_id, new_id in RENAME.items():
        if old_id not in existing:
            print(f"  WARN  RENAME source not found: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  RENAME  {old_id!r} -> {new_id!r}  ({n}r)")
        if write:
            m.loc[m["model_id"]   == old_id, "model_id"]   = new_id
            m.loc[m["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_id"]   == old_id, "model_id"]   = new_id
            existing = set(m["model_id"])
            rename_count += 1

    # ── 2. REMAP ──────────────────────────────────────────────────────────────
    remap_count = 0
    for old_id, new_id in REMAP.items():
        cur = set(m["model_id"]) if write else existing | set(RENAME.values())
        if old_id not in (set(m["model_id"]) if write else set(m_df["model_id"])):
            print(f"  WARN  REMAP source not found: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  REMAP  {old_id!r} -> {new_id!r}  ({n}r)")
        if write:
            r.loc[r["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_id"]   == old_id, "model_id"]   = new_id
            m = m[m["model_id"] != old_id]
            remap_count += 1

    # ── 3. Post-merge dedup ───────────────────────────────────────────────────
    dupes_dropped = 0
    if write:
        key_cols = [c for c in config.RESULT_IDENTITY_KEY if c in r.columns]
        before = len(r)
        r = r.drop_duplicates(subset=key_cols).reset_index(drop=True)
        dupes_dropped = before - len(r)

    return m, r, rename_count, remap_count, dupes_dropped


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
    m_new, r_new, n_rename, n_remap, n_dupes = apply(m_df, r_df, write=args.write)

    if args.write:
        io.save_csv(m_new, config.MODELS_CSV)
        io.save_results(r_new)
        print(f"\nWritten.")
        print(f"  models:  {len(m_df)} → {len(m_new)}  (−{len(m_df)-len(m_new)})")
        print(f"  results: {len(r_df)} → {len(r_new)}  (−{len(r_df)-len(r_new)}, "
              f"{n_dupes} post-merge dupes dropped)")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        print(f"\nDry run summary:")
        print(f"  RENAME: {len(RENAME)}  REMAP: {len(REMAP)}")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
