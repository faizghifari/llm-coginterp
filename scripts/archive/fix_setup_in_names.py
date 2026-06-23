#!/usr/bin/env python3
"""
Extract Setup Qualifiers from Model Names
==========================================
The methodology states: "Models that represent a different *setup* of the same
model (e.g., different context length, CoT prompting, effort levels) are captured
via the `setup` and `reasoning_enabled` columns instead of as separate model
entries."

This script finds every model whose ID encodes a setup qualifier in a trailing
parenthetical (zero-shot, few-shot, fine-tuned, CoT, greedy, etc.) and:

  1. Moves the qualifier into results.setup for every result row of that model.
  2. Renames the model to its base ID (qualifier stripped).
     - If the base model already exists → REMAP (repoint results, delete old entry).
     - If the base model is new → RENAME (update the model entry in-place).

Exceptions:
  - CLIP variants and BiDAF variants are CASCADE-REMOVED — they are not generative
    language models and do not belong in the dataset.
  - Any entry in SKIP_BASES is not auto-processed (handled elsewhere or intentional).

Also handles encoder-only model removals that belong in this pass:
  BERT large (LAMB optimizer), PromptNER [BERT-large], PromptNER [RoBERTa-large],
  FLERT XLM-R, XLM-R (encoder-only, excluded by methodology).
  MiniGPT-4-7B (BERTScore) → MiniGPT-4-7B (metric name in model ID, not a model).

Usage:
  python3 scripts/fix_setup_in_names.py           # dry run
  python3 scripts/fix_setup_in_names.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse, re, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scripts.lib import io, config

# ─────────────────────────────────────────────────────────────────────────────
# Setup detection regex — matches a trailing parenthetical containing a setup
# keyword.  The base model name is group(1); the setup value is group(2).
# ─────────────────────────────────────────────────────────────────────────────
SETUP_RE = re.compile(
    r'^(.*?)\s*\(('
    r'zero[\-\s]shot'
    r'|0[\-\s]shot'
    r'|few[\-\s]shot'
    r'|\d+[\-\s]shot'
    r'|one[\-\s]shot'
    r'|fine[\-\s]tun\w*'
    r'|finetuned'
    r'|prompt[\-\s]tuned'
    r'|greedy.*'
    r'|0-shot cot'
    r'|.*\bcot\b.*'
    r'|\d+-shot.*cot'
    r'|.*chain[^\)]*thought.*'
    r'|maj\d+@\d+.*'
    r'|logit[\s\-]scoring'
    r'|rank[\s\-]class.*'
    r')\)\s*$',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# CASCADE REMOVE — these model IDs are not generative LLMs (encoder-only or
# task-specific discriminative models).  Deleted along with all their results.
# ─────────────────────────────────────────────────────────────────────────────
ENCODER_REMOVE = {
    # Encoder-only models (cannot generate arbitrary text)
    "BERT large (LAMB optimizer)":     "encoder-only (BERT)",
    "PromptNER [BERT-large]":          "encoder-only (BERT) NER model",
    "PromptNER [RoBERTa-large]":       "encoder-only (RoBERTa) NER model",
    "FLERT XLM-R":                     "encoder-only (XLM-R) fine-tuned for NER",
    "XLM-R":                           "encoder-only (XLM-R)",
    # Not generative LLMs — appear only as setup variants
    "CLIP (finetuned)":                "CLIP is a vision-language embedding model, not a generative LLM",
    "CLIP-RN50 (Zero-Shot)":           "CLIP is a vision-language embedding model, not a generative LLM",
    "CLIP-RN50x64/14 (Zero-Shot)":     "CLIP is a vision-language embedding model, not a generative LLM",
    "CLIP-ViL (Zero-Shot)":            "CLIP-ViL is a vision-language model, not a generative LLM",
    "CLIP-ViT-B/32 (Zero-Shot)":       "CLIP is a vision-language embedding model, not a generative LLM",
    "CLIP-ViT-L/14 (Zero-Shot)":       "CLIP is a vision-language embedding model, not a generative LLM",
    "BiDAF + ELMo (fine-tuned)":       "BiDAF is a reading comprehension model, not a generative LLM",
    "BiDAF-MultiNLI (fine-tuned)":     "BiDAF is a reading comprehension model, not a generative LLM",
}

# ─────────────────────────────────────────────────────────────────────────────
# RENAME — standalone fixes that don't fit the auto-detection pattern
# ─────────────────────────────────────────────────────────────────────────────
EXTRA_RENAME = {
    # "(BERTScore)" is an evaluation metric appended to the model name, not the model
    "MiniGPT-4-7B (BERTScore)": "MiniGPT-4-7B",
}

# ─────────────────────────────────────────────────────────────────────────────
# SKIP_BASES — base model names that should NOT be auto-processed even if the
# regex would match them.  Add entries here for models where the parenthetical
# is genuinely part of the model identity (e.g., architectural variant).
# ─────────────────────────────────────────────────────────────────────────────
SKIP_BASES: set = set()

# ─────────────────────────────────────────────────────────────────────────────
# SETUP_NORMALISE — map raw extracted strings to canonical display values
# ─────────────────────────────────────────────────────────────────────────────
_NORM = {
    "zero-shot": "Zero-shot", "zero shot": "Zero-shot",
    "0-shot": "0-shot",
    "few-shot": "Few-shot", "few shot": "Few-shot",
    "one-shot": "1-shot",
    "finetuned": "Fine-tuned", "fine-tuned": "Fine-tuned",
    "fine tuned": "Fine-tuned",
    "prompt-tuned": "Prompt-tuned", "prompt tuned": "Prompt-tuned",
}

def normalise_setup(raw: str) -> str:
    key = raw.lower().strip()
    return _NORM.get(key, raw.strip())


def detect_setup_variants(model_ids: set) -> list[tuple[str, str, str]]:
    """Return list of (old_id, base_id, setup_val) for every auto-detected entry."""
    entries = []
    for mid in sorted(model_ids):
        if mid in ENCODER_REMOVE:
            continue
        m = SETUP_RE.search(mid)
        if not m:
            continue
        base = m.group(1).strip()
        setup = normalise_setup(m.group(2))
        if base in SKIP_BASES:
            continue
        entries.append((mid, base, setup))
    return entries


def apply_setup_extract(m_df: pd.DataFrame, r_df: pd.DataFrame, write: bool):
    m = m_df.copy()
    r = r_df.copy()
    existing_ids = set(m["model_id"])

    # ── 1. Encoder-only / non-LLM removes ────────────────────────────────────
    remove_n = 0
    for mid, reason in ENCODER_REMOVE.items():
        if mid not in existing_ids:
            print(f"  WARN  REMOVE target not found (already gone?): {mid!r}")
            continue
        n = (r["model_name"] == mid).sum()
        print(f"  REMOVE  {mid!r}  ({n} results)  — {reason}")
        if write:
            m = m[m["model_id"] != mid]
            r = r[r["model_name"] != mid]
        remove_n += 1

    # ── 2. Extra renames (non-regex) ─────────────────────────────────────────
    for old_id, new_id in EXTRA_RENAME.items():
        if old_id not in existing_ids:
            print(f"  WARN  RENAME target not found (already gone?): {old_id!r}")
            continue
        if new_id in existing_ids:
            # target exists already → REMAP
            n = (r["model_name"] == old_id).sum()
            print(f"  RENAME→REMAP  {old_id!r} -> {new_id!r}  ({n} results)")
            if write:
                r.loc[r["model_name"] == old_id, "model_name"] = new_id
                r.loc[r["model_id"]   == old_id, "model_id"]   = new_id
                m = m[m["model_id"] != old_id]
        else:
            n = (r["model_name"] == old_id).sum()
            print(f"  RENAME  {old_id!r} -> {new_id!r}  ({n} results)")
            if write:
                m.loc[m["model_id"]   == old_id, "model_id"]   = new_id
                m.loc[m["model_name"] == old_id, "model_name"] = new_id
                r.loc[r["model_name"] == old_id, "model_name"] = new_id
                r.loc[r["model_id"]   == old_id, "model_id"]   = new_id

    # ── 3. Setup extraction ───────────────────────────────────────────────────
    # Recompute existing_ids after removes + renames
    if write:
        existing_ids = set(m["model_id"])
    else:
        existing_ids = set(m_df["model_id"]) - set(ENCODER_REMOVE) | set(EXTRA_RENAME.values())

    entries = detect_setup_variants(existing_ids)

    # Group by base_id to report collisions cleanly
    by_base: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for old_id, base_id, setup_val in entries:
        by_base[base_id].append((old_id, setup_val))

    remap_count = rename_count = 0
    for base_id, variants in sorted(by_base.items()):
        base_exists = base_id in existing_ids
        op = "REMAP" if base_exists else "RENAME→CREATE"
        for old_id, setup_val in variants:
            n_results = (r["model_name"] == old_id).sum() if not write else 0
            if not write:
                n_results = (r_df["model_name"] == old_id).sum()
                print(f"  {op}  {old_id!r}")
                print(f"       -> base={base_id!r}  setup={setup_val!r}  ({n_results} results)")
            if write:
                # a. Move setup qualifier into results.setup
                mask = r["model_name"] == old_id
                needs_setup = mask & (r["setup"] == "")
                r.loc[needs_setup, "setup"] = setup_val
                # b. Update model_name (FK) and model_id in results
                r.loc[mask, "model_name"] = base_id
                r.loc[r["model_id"] == old_id, "model_id"] = base_id
                # c. Handle model entry
                if base_id in set(m["model_id"]):
                    # REMAP: base exists, delete old entry
                    m = m[m["model_id"] != old_id]
                    remap_count += 1
                else:
                    # RENAME: update old entry to base_id (first variant creates it)
                    m.loc[m["model_id"] == old_id, "model_id"]   = base_id
                    m.loc[m["model_name"] == old_id, "model_name"] = base_id
                    existing_ids = set(m["model_id"])  # refresh after rename
                    rename_count += 1

    # ── 4. Post-merge dedup on identity key ──────────────────────────────────
    dupes_dropped = 0
    if write:
        id_key = config.RESULT_IDENTITY_KEY
        # Use a subset that's always populated
        key_cols = [c for c in id_key if c in r.columns]
        before = len(r)
        r = r.drop_duplicates(subset=key_cols).reset_index(drop=True)
        dupes_dropped = before - len(r)

    return m, r, remove_n, remap_count, rename_count, dupes_dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    _, m_df, r_df = io.load_data()
    print(f"  models: {len(m_df)}  results: {len(r_df)}")

    print(f"\n{'Applying' if args.write else 'Dry run —'} changes:\n")
    m_new, r_new, n_rem, n_remap, n_rename, n_dupes = apply_setup_extract(
        m_df, r_df, write=args.write
    )

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
        entries = detect_setup_variants(set(m_df["model_id"]))
        by_base = defaultdict(list)
        for old_id, base_id, setup_val in entries:
            by_base[base_id].append(old_id)
        remap_c  = sum(1 for b in by_base if b in set(m_df["model_id"]))
        rename_c = sum(1 for b in by_base if b not in set(m_df["model_id"]))
        print(f"\nDry run summary:")
        print(f"  Encoder-only / non-LLM removes: {len(ENCODER_REMOVE)}")
        print(f"  Extra renames:                  {len(EXTRA_RENAME)}")
        print(f"  Setup-variant → base models:")
        print(f"    REMAP  (base already exists): {sum(len(v) for b, v in by_base.items() if b in set(m_df['model_id']))} models across {remap_c} bases")
        print(f"    RENAME (base is new):         {sum(len(v) for b, v in by_base.items() if b not in set(m_df['model_id']))} models across {rename_c} bases")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
