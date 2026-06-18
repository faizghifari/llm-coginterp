#!/usr/bin/env python3
"""
Model Deduplication and Standardisation
========================================
Cleans models.csv and cascades all changes to results.csv.

Three operations:
  REMOVE  — delete model row AND all its result rows (non-model entries, baselines)
  REMAP   — redirect all result rows from an old model_id to an existing canonical one,
            then delete the old model row (duplicate under a different name)
  RENAME  — change a model's own ID everywhere (HF repo-path cleanup, newline fixes)
            while keeping all its result rows

Usage:
  python3 scripts/deduplicate_models.py           # dry run (print plan, no changes)
  python3 scripts/deduplicate_models.py --write   # apply changes

Always run  python3 scripts/verify_data.py  and
            python3 scripts/manage_data.py recompute-stats --write
afterwards.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scripts.lib import io, config

# ═══════════════════════════════════════════════════════════════════════════════
# REMOVE — cascade-delete model + all its results
# Reason documented in each entry.
# ═══════════════════════════════════════════════════════════════════════════════
REMOVE = {
    "Random":                                       "not a model — random baseline",
    "epoch 9 pgd_25_0.1_eps":                      "adversarial attack training checkpoint, not a model",
    "TinyLlama-1.1B-intermediate-step-1195k-token-2.5T": "training checkpoint (step), not a release",
    "TinyLlama-1.1B-intermediate-step-1431k-3T":   "training checkpoint (step), not a release",
    "TinyLlama-1.1B-intermediate-step-955k-token-2T":   "training checkpoint (step), not a release",
    "tensorflow/tensor2tensor":                     "ML framework, not a model",
    "Zero-shot":                                    "describes a setup, not a model",
    "Zero Shot":                                    "describes a setup, not a model",
}

# ═══════════════════════════════════════════════════════════════════════════════
# REMAP — old_id -> canonical_id (canonical already exists in models.csv)
# All result rows for old_id are repointed to canonical_id.
# old_id model row is then deleted.
# ═══════════════════════════════════════════════════════════════════════════════
REMAP = {
    # ── HF repo-path → existing canonical ────────────────────────────────────
    "OpenAI/GPT-4o":              "GPT-4o",
    "Anthropic/claude-3-7-sonnet":"Claude 3.7 Sonnet",
    "OpenAI/o3-mini":             "o3-mini-2025-01-31",
    "xAI/grok-3-1212":            "Grok-3",
    "OpenAI/o1-2024-12-17-high":  "o1-2024-12-17",   # -high = reasoning_effort=high (setup, not a new model)
    # ── Dedup: lowercase/hyphen variants → canonical ─────────────────────────
    "GPT OSS 120B":               "gpt-oss-120b",
    "gpt-j-6b":                   "GPT-J (6B)",
    "LLaMA-2-70B":                "Llama 2 (70B)",
    "gpt-neox-20b":               "GPT-NeoX (20B)",
    "falcon-40b":                 "Falcon (40B)",
    "Gemini-2.0-Flash-Lite":      "Gemini 2.0 Flash Lite",
    "falcon-7b":                  "Falcon (7B)",
    "opt-66b":                    "OPT (66B)",
    "LLaMA-2-7B":                 "Llama 2 (7B)",
    "LLaMa-2-13b":                "Llama 2 (13B)",
    "GPT-3.5 Turbo (0613)":       "gpt-3.5-turbo-0613",
    "mpt-30b":                    "MPT (30B)",
    "T5-Base":                    "T5-base",
    "pythia-12b":                 "Pythia (12B)",
    "pythia-6.9b":                "Pythia (6.9B)",
    "dbrx-instruct":              "DBRX Instruct",
    "Command-R":                  "Command R",
    "Yi (34B)":                   "Yi-34B",
    "Yi (6B)":                    "Yi-6B",
    "gemma-7b":                   "Gemma (7B)",
    "fastText":                   "FastText",
    "GPT-3 (Zero-Shot)":          "GPT-3 (zero-shot)",
    "GPT-3 zero-shot":            "GPT-3 (zero-shot)",
    "mistral-large-2402":         "Mistral Large (2402)",
    "DECAPROP":                   "DecaProp",
    "SAFSr_model":                "SAFSR model",
    "Pegasus":                    "PEGASUS",
    "mistral-large-2411":         "Mistral Large (2411)",
    "seq2seq":                    "Seq2Seq",
    "Qwen3.5-9B":                 "Qwen3.5 9B",
    "GPT-3-175B (Few-Shot)":      "GPT-3 175B (Few-Shot)",
    "GPT-3-175B (few-shot)":      "GPT-3 175B (Few-Shot)",
    "GPT-2 (1.5B)":               "GPT-2 1.5B",
    "GPT-2 Large":                "gpt2-large",
    "GPT-2 (large)":              "gpt2-large",
    "GPT-2 Medium":               "gpt2-medium",
    "GPT-2 (medium)":             "gpt2-medium",
    "GPT-3 (175B)":               "GPT-3 175B",
    "ST_VQA":                     "ST-VQA",
    "Lead-3":                     "LEAD-3",
    "UNITER large":               "UNITER (Large)",
    "WizardCoder-15b":            "WizardCoder-15B",
    "WizardCoder 15B":            "WizardCoder-15B",
    "PaLM 2(few-shot, k=3, CoT)": "PaLM 2 (few-shot, k=3, CoT)",
    "ULMFit":                     "ULMFiT",
    "pythia-1b":                  "Pythia (1B)",
    "PaLM 540B (zero-shot)":      "PaLM-540B (Zero-Shot)",
    "ALBEF 14M":                  "ALBEF (14M)",
    "Grok-3-Mini-beta":           "Grok 3 mini Beta",
    "CRONKGQA":                   "CronKGQA",
    "OPT 175B (50% Sparsity)":    "OPT-175B (50% Sparsity)",
    "PEGASUS 2B + SliC":          "PEGASUS 2B + SLiC",
    "Transformer+Wdrop":          "Transformer+WDrop",
    "GPT-3-175B (Zero-Shot)":     "GPT-3 175B (Zero-Shot)",
    "GPT-3 175B (zero-shot)":     "GPT-3 175B (Zero-Shot)",
    "Neo-6B (Few-Shot)":          "Neo-6B (few-shot)",
    "Unik-Qa":                    "UniK-QA",
    "mPLUG (Huge)":               "mPLUG-Huge",
    "Tranx":                      "TranX",
    "MoTCoder-7B-v1.5":           "MoTCoder-7B-V1.5",
    "Deepstruct multi-task":      "DeepStruct multi-task",
    "ABS":                        "Abs",
    "PaLM 540B (few-shot)":       "PaLM-540B (Few-Shot)",
    "SLQA(ensemble)":             "SLQA (ensemble)",
    "MEMEN  (single model)":      "MEMEN (single model)",
    "FPNet  (ensemble)":          "FPNet (ensemble)",
    "{MTL} (single model)":       "MTL (single model)",
    "Hanvon_model(single model)": "Hanvon_model (single model)",
    "synss (single model )":      "synss (single model)",
    "ICL_MODEL(ensemble)":        "ICL_MODEL (ensemble)",
    "BLIP 129M":                  "BLIP-129M",
    "Primer":                     "PRIMER",
    "Blockwise(baseline)":        "Blockwise (baseline)",
    "GPT-3 175B (1 shot)":        "GPT-3 175B (1-shot)",
    "GLaM 62B/64E (One-shot)":    "GLaM 62B/64E (One-Shot)",
    "GLaM 62B/64E (Zero-shot)":   "GLaM 62B/64E (Zero-Shot)",
    "Llama 2 34B (0-shot)":       "LLaMA 2 34B (0-shot)",
    "Llama 2 13B (0-shot)":       "LLaMA 2 13B (0-shot)",
    "Snorkel MeTaL(ensemble)":    "Snorkel MeTaL (ensemble)",
    "Recurrent highway networks": "Recurrent Highway Networks",
    "Hybrid H3 (125M)":           "Hybrid H3 125M",
    "Transformer  (Adaptive inputs)": "Transformer (Adaptive inputs)",
    "All-attention network - 36 layers": "All-attention network (36 layers)",
    "Binder":                     "BINDER",
    "Unidrop":                    "UniDrop",
    "LLaMA-3 8B+MoSLoRA (fine-tuned)": "LLaMA 3 8B+MoSLoRA (fine-tuned)",
    "Chinchilla (Zero-Shot)":     "Chinchilla (zero-shot)",
    "code-davinci-002 175B + REPLUG LSR (Few-Shot)": "code-davinci-002 175B + REPLUG LSR (few-shot)",
    "code-davinci-002 175B + REPLUG (Few-Shot)":     "code-davinci-002 175B + REPLUG (few-shot)",
    "ChatQA-1.5-llama3-8b (Zero-Shot, KILT)": "ChatQA-1.5-llama3-8B (Zero-Shot, KILT)",
    "GLaM 62B/64E (Few-shot)":    "GLaM 62B/64E (Few-Shot)",
    "GPT-3-175B (One-Shot)":      "GPT-3 175B (one-shot)",
    "SparseGPT 175B (50% sparsity)": "SparseGPT 175B (50% Sparsity)",
    "SparseGPT 175B (2:4 sparsity)": "SparseGPT 175B (2:4 Sparsity)",
    "Llama 2 7B (0-shot)":        "LLaMA 2 7B (0-shot)",
    "phi-1.5-web 1.3B":           "phi-1.5-web (1.3B)",
    "Exaqt":                      "EXAQT",
    "Explaignn":                  "EXPLAIGNN",
    "Uniqorn":                    "UniQorn",
    "InstructGpt":                "InstructGPT",
    "CfC":                        "CFC",
    "Gemini-Pro 4-shot":          "Gemini Pro (4-shot)",
    "GLaM 64B/64E (0 shot)":      "GLaM 64B/64E (0-shot)",
    "GPT-2 Small":                "GPT-2 (small)",
    "Transformer-XL - 24 layers": "Transformer-XL (24 layers)",
    "All-attention network - 18 layers": "All-attention network (18 layers)",
    "BP-Transformer - 12 Layers": "BP-Transformer (12 layers)",
    "Cohere  Large":              "Cohere Large",
    "OllIE Mausam et al. (2012)": "Ollie Mausam et al. (2012)",
    "GPT-3.5 turbo (175B)":       "GPT-3.5 Turbo (175B)",
    "MuggleMATH-70B":             "MuggleMATH 70B",
    "MuggleMATH-13B":             "MuggleMATH 13B",
    "LLaMA 65B-maj1@k":           "LLaMA 65B (maj1@k)",
    "LLaMA 7B-maj1@k":            "LLaMA 7B (maj1@k)",
    # newline → space merges (clean version already exists)
    "resnet8x4\n(T: resnet32x4 S: resnet8x4)":        "resnet8x4 (T: resnet32x4 S: resnet8x4)",
    "DePlot+FlanPaLM+Codex\n(PoT Self-Consistency)":   "DePlot+FlanPaLM+Codex (PoT Self-Consistency)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# RENAME — change a model's own ID in-place (old_id doesn't duplicate anything)
# The model row stays; model_id, model_name, and all result rows are updated.
# ═══════════════════════════════════════════════════════════════════════════════
RENAME = {
    # HF repo-path → clean canonical name (no existing canonical in models.csv)
    "OpenAI/o3-2025-01-31-high":     "o3-2025-01-31-high",
    "OpenAI/o4-mini-2025-05-01-high":"o4-mini-2025-05-01-high",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek-Coder-6.7B-Instruct",
    "Riple/Saanvi-v0.5-DeepAnalysis": "Saanvi-v0.5-DeepAnalysis",
    "Riple/Saanvi-v0.1":              "Saanvi-v0.1",
    # Newline fixes (clean version doesn't exist yet)
    "Self-Evaluation Guided Decoding\n(Codex, CoT, single reasoning chain, 6-shot gen, 4-shot eval)":
        "Self-Evaluation Guided Decoding (Codex, CoT, single reasoning chain, 6-shot gen, 4-shot eval)",
    "Qwen2-Math-72B-Instruct\n(greedy)": "Qwen2-Math-72B-Instruct (greedy)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════════

def validate_maps(m_df):
    existing = set(m_df["model_id"])
    ok = True
    for old_id in REMAP:
        canonical = REMAP[old_id]
        if old_id not in existing:
            print(f"  WARN REMAP source not in models.csv (already gone?): {old_id!r}")
        if canonical not in existing:
            print(f"  ERROR REMAP target not in models.csv: {old_id!r} -> {canonical!r}")
            ok = False
    for old_id in REMOVE:
        if old_id not in existing:
            print(f"  WARN REMOVE target not in models.csv (already gone?): {old_id!r}")
    for old_id, new_id in RENAME.items():
        if old_id not in existing:
            print(f"  WARN RENAME source not in models.csv (already gone?): {old_id!r}")
        if new_id in existing:
            print(f"  WARN RENAME target already exists — will REMAP instead: {old_id!r} -> {new_id!r}")
    return ok


def apply(m_df, r_df, write=False):
    m = m_df.copy()
    r = r_df.copy()
    existing = set(m["model_id"])

    removes = 0; remap_rows = 0; rename_rows = 0

    # ── REMOVE ────────────────────────────────────────────────────────────────
    for mid, reason in REMOVE.items():
        if mid not in existing:
            continue
        n = (r["model_name"] == mid).sum()
        if not write:
            print(f"  REMOVE {mid!r}  ({n} results)  — {reason}")
        else:
            m = m[m["model_id"] != mid]
            r = r[r["model_name"] != mid]
        removes += 1
        remap_rows += n

    # ── REMAP ─────────────────────────────────────────────────────────────────
    for old_id, canonical in REMAP.items():
        if old_id not in set(m["model_id"]):
            continue
        n = (r["model_name"] == old_id).sum()
        if not write:
            print(f"  REMAP  {old_id!r} -> {canonical!r}  ({n} results)")
        else:
            r.loc[r["model_name"] == old_id, "model_name"] = canonical
            r.loc[r["model_id"]   == old_id, "model_id"]   = canonical
            m = m[m["model_id"] != old_id]
        remap_rows += n

    # ── RENAME ────────────────────────────────────────────────────────────────
    for old_id, new_id in RENAME.items():
        if old_id not in set(m["model_id"]):
            continue
        # if new_id already exists, treat as REMAP
        if new_id in set(m["model_id"]):
            n = (r["model_name"] == old_id).sum()
            if not write:
                print(f"  RENAME->REMAP {old_id!r} -> {new_id!r}  ({n} results, target exists)")
            else:
                r.loc[r["model_name"] == old_id, "model_name"] = new_id
                r.loc[r["model_id"]   == old_id, "model_id"]   = new_id
                m = m[m["model_id"] != old_id]
            remap_rows += n
        else:
            n = (r["model_name"] == old_id).sum()
            if not write:
                print(f"  RENAME {old_id!r} -> {new_id!r}  ({n} results)")
            else:
                m.loc[m["model_id"]   == old_id, "model_id"]   = new_id
                m.loc[m["model_name"] == old_id, "model_name"] = new_id
                r.loc[r["model_name"] == old_id, "model_name"] = new_id
                r.loc[r["model_id"]   == old_id, "model_id"]   = new_id
            rename_rows += n

    return m, r, removes, remap_rows, rename_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    main_b, m_df, r_df = io.load_data()
    print(f"  models: {len(m_df)}  results: {len(r_df)}")

    print("\nValidating maps...")
    if not validate_maps(m_df):
        sys.exit(1)

    print(f"\n{'Applying' if args.write else 'Dry run —'} changes:")
    m_new, r_new, n_removes, n_remap, n_rename = apply(m_df, r_df, write=args.write)

    if args.write:
        # Dedup results that became duplicates after merging
        id_key = ["benchmark_id", "model_name", "metric_name", "source_url"]
        before = len(r_new)
        r_new = r_new.drop_duplicates(subset=id_key).reset_index(drop=True)
        deduped = before - len(r_new)

        io.save_csv(m_new, config.MODELS_CSV)
        io.save_results(r_new)
        print(f"\nWritten.")
        print(f"  models:  {len(m_df)} → {len(m_new)}  (−{len(m_df)-len(m_new)})")
        print(f"  results: {len(r_df)} → {len(r_new)}  (−{len(r_df)-len(r_new)}, {deduped} post-merge dupes dropped)")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        print(f"\nDry run summary:")
        print(f"  Models to remove:             {n_removes}")
        print(f"  Models to remap (merge/fix):  {len(REMAP)}")
        print(f"  Models to rename (ID fix):    {len(RENAME)}")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
