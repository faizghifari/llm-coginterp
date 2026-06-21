#!/usr/bin/env python3
"""
Standardisation Pass 7 — Remaining Task-Specific Stragglers
============================================================
Removes 9 task-specific / discriminative models that slipped through pass 6
because they lacked the `+` compound-name pattern the earlier sweep relied on.
All fail the scope rule: "general-purpose generative model that can take and
process any arbitrary text and task."

Operations:
  REMOVE — cascade delete model + all results rows

Usage:
  python3 scripts/standardise_pass7.py           # dry run
  python3 scripts/standardise_pass7.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io, config

# ─────────────────────────────────────────────────────────────────────────────
# REMOVE — not general-purpose generative; cascade delete model + results.
# ─────────────────────────────────────────────────────────────────────────────
REMOVE = {
    # Discriminative graph-based QA / retrieval readers
    "GRAFT-Net":
        "discriminative graph QA reader (KGQA, not generative)",
    "Robustly Fine-tuned Graph-based Recurrent Retriever":
        "discriminative graph+retrieval reader (HotpotQA, not generative)",

    # Entity alignment graph neural networks — not generative at all
    "GCN-Align":
        "entity alignment GCN (not a generative LM)",
    "AlignEA":
        "entity alignment model (not a generative LM)",

    # Discriminative ASR / speech classifier
    "Pooling classifier pre-trained using force-aligned phoneme and word labels on LibriSpeech":
        "discriminative speech/ASR classifier (not a generative LM)",

    # Discriminative NLI classifier
    "Fine-Tuned LM-Pretrained Transformer":
        "NLI-only fine-tune on SNLI (discriminative classifier, not generative)",

    # Multi-agent compound system (not a single model)
    "ChatDev":
        "multi-agent compound system (software dev pipeline, not a single model)",

    # Task-specific fine-tunes with no general-purpose capability
    "InstructABSA":
        "task-specific ABSA fine-tune (SemEval 2014 sentiment only)",
    "InstructDS":
        "task-specific dialogue summarisation fine-tune (SAMSum/DialogSum only)",
}


def validate(m_df):
    existing = set(m_df["model_id"])
    ok = True
    for model_id in REMOVE:
        if model_id not in existing:
            print(f"  WARN  REMOVE source missing: {model_id!r}")
    return ok


def apply(m_df, r_df, write: bool):
    m = m_df.copy()
    r = r_df.copy()

    for model_id, reason in REMOVE.items():
        if model_id not in set(m["model_id"]):
            print(f"  WARN  REMOVE source missing: {model_id!r}")
            continue
        n = (r_df["model_name"] == model_id).sum()
        print(f"  REMOVE  {model_id!r}  ({n}r)  [{reason}]")
        if write:
            r = r[r["model_name"] != model_id]
            m = m[m["model_id"]   != model_id]

    return m, r


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    _, m_df, r_df = io.load_data()
    print(f"  models: {len(m_df)}  results: {len(r_df)}")

    print("\nValidating maps...")
    validate(m_df)
    print("  All sources verified ✓")

    print(f"\n{'Applying' if args.write else 'Dry run —'} changes:\n")
    m_new, r_new = apply(m_df, r_df, write=args.write)

    if args.write:
        io.save_csv(m_new, config.MODELS_CSV)
        io.save_results(r_new)
        print(f"\nWritten.")
        print(f"  models:  {len(m_df)} → {len(m_new)}  (−{len(m_df)-len(m_new)})")
        print(f"  results: {len(r_df)} → {len(r_new)}  (−{len(r_df)-len(r_new)})")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        print(f"\nDry run summary:")
        print(f"  REMOVE: {len(REMOVE)}")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
