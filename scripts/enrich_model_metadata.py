#!/usr/bin/env python3
"""
Metadata Enrichment — categorize-models flagged entries
========================================================
Fills `developer`, `model_family`, and `base_model` for 108 valid general-purpose
models that were flagged by `manage_data.py categorize-models` solely because those
fields were blank (not because the models are out of scope).

Sources: HuggingFace Hub lookups + training-data knowledge.
Fields: only blank fields are written; non-blank values are not overwritten,
        EXCEPT `developer` where the stored HF org slug is upgraded to a
        proper display name (e.g. "wizardlmteam" → "Microsoft").

Usage:
  python3 scripts/enrich_model_metadata.py           # dry run
  python3 scripts/enrich_model_metadata.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io, config

# ─────────────────────────────────────────────────────────────────────────────
# ENRICH — model_id → (developer, model_family, base_model)
#   "" = leave blank (unknown / not applicable)
#   Non-blank values are applied even if a different value was already stored,
#   ONLY for the developer field (casing/slug upgrades). For family/base, we
#   only fill empty cells.
# ─────────────────────────────────────────────────────────────────────────────
ENRICH = {
    # ── OpenAI ─────────────────────────────────────────────────────────────────
    "ChatGPT":                              ("OpenAI", "GPT-3.5", "GPT-3.5-Turbo"),
    "ChatGPT 3.5 SpatialFormat":            ("OpenAI", "GPT-3.5", "GPT-3.5-Turbo"),
    "ChatGPT w/ tkg":                       ("OpenAI", "GPT-3.5", "GPT-3.5-Turbo"),
    "ChatGPT w/o tkg":                      ("OpenAI", "GPT-3.5", "GPT-3.5-Turbo"),
    "ChatGPT (gpt-3.5-turbo, few-shot)":    ("OpenAI", "GPT-3.5", "GPT-3.5-Turbo"),
    "ChatGPT (gpt-3.5-turbo, zero-shot)":   ("OpenAI", "GPT-3.5", "GPT-3.5-Turbo"),
    "ChatGPT (Ask, Refine, Trust)":         ("OpenAI", "GPT-3.5", "GPT-3.5-Turbo"),
    "InstructGPT":                          ("OpenAI", "GPT-3", "GPT-3 (175B)"),
    "GPT-4 (RLHF)":                         ("OpenAI", "GPT-4", "GPT-4"),
    "GPT-4 (ChatGPT Plus)":                 ("OpenAI", "GPT-4", "GPT-4"),
    "GPT-4 (Bing Chat)":                    ("OpenAI", "GPT-4", "GPT-4"),
    "GPT-4o-mini Fine-Tuned":               ("OpenAI", "GPT-4o", "GPT-4o mini"),
    "GPT-4o Fine-Tuned (Minimal)":          ("OpenAI", "GPT-4o", "GPT-4o"),
    "GPT-2-Medium 355M (fine-tuned, BS=5)": ("OpenAI", "GPT-2", "GPT-2 Medium (355M)"),

    # ── Google ─────────────────────────────────────────────────────────────────
    "chat-bison":     ("Google", "PaLM 2", "PaLM 2"),
    "codechat-bison": ("Google", "PaLM 2", "PaLM 2"),

    # ── Hugging Face ───────────────────────────────────────────────────────────
    "starchat-alpha":             ("Hugging Face", "StarCoder", "StarCoder (15.5B)"),
    "starchat-beta":              ("Hugging Face", "StarCoder", "StarCoder (15.5B)"),
    "zephyr-7b-alpha":            ("Hugging Face", "Mistral",   "Mistral v0.1 (7B)"),
    "zephyr-7b-beta":             ("Hugging Face", "Mistral",   "Mistral v0.1 (7B)"),
    "zephyr-orpo-141b-A35b-v0.1": ("Hugging Face", "Mixtral",   "Mixtral-8x22B-v0.1"),
    "IDEFICS-instruct (9B)":      ("Hugging Face", "IDEFICS",   "IDEFICS (9B)"),
    "IDEFICS-instruct (80B)":     ("Hugging Face", "IDEFICS",   "IDEFICS (80B)"),

    # ── Intel ──────────────────────────────────────────────────────────────────
    "neural-chat-7b-v3":         ("Intel", "Mistral", "Mistral v0.1 (7B)"),
    "neural-chat-7b-v3-1":       ("Intel", "Mistral", "Mistral v0.1 (7B)"),
    "neural-chat-7b-v3-2":       ("Intel", "Mistral", "Mistral v0.1 (7B)"),
    "neural-chat-7b-v3-3":       ("Intel", "Mistral", "Mistral v0.1 (7B)"),
    "neural-chat-7b-v3-3-Slerp": ("Intel", "Mistral", "Mistral v0.1 (7B)"),

    # ── LLM360 ─────────────────────────────────────────────────────────────────
    "AmberChat": ("LLM360", "Amber", "Amber (LLM360)"),
    "K2-Chat":   ("LLM360", "K2",    "K2 (LLM360)"),

    # ── Nous Research ──────────────────────────────────────────────────────────
    "Nous-Hermes-13b":           ("Nous Research", "LLaMA",  "LLaMA 13B"),
    "Nous-Hermes-2-SOLAR-10.7B": ("Nous Research", "SOLAR",  "SOLAR (10.7B)"),
    "Yarn-Solar-10b-32k":        ("Nous Research", "SOLAR",  "SOLAR (10.7B)"),
    "Yarn-Solar-10b-64k":        ("Nous Research", "SOLAR",  "SOLAR (10.7B)"),

    # ── Open-Orca ──────────────────────────────────────────────────────────────
    "LlongOrca-13B-16k":              ("Open-Orca", "LLaMA 2", "Llama-2-13B"),
    "LlongOrca-7B-16k":               ("Open-Orca", "LLaMA 2", "Llama-2-7B"),
    "OpenOrca-Platypus2-13B":         ("Open-Orca", "LLaMA 2", "Llama-2-13B"),
    "OpenOrca-Preview1-13B":          ("Open-Orca", "LLaMA",   "LLaMA 13B"),
    "OpenOrcaxOpenChat-Preview2-13B": ("Open-Orca", "LLaMA 2", "Llama-2-13B"),

    # ── PrimeIntellect ─────────────────────────────────────────────────────────
    "INTELLECT-1-Instruct": ("PrimeIntellect", "INTELLECT", "INTELLECT-1"),

    # ── Microsoft ──────────────────────────────────────────────────────────────
    # (developer upgraded from HF slug "wizardlmteam")
    "WizardLM-13B-V1.0":  ("Microsoft", "LLaMA",      "LLaMA 13B"),
    "WizardLM-13B-V1.2":  ("Microsoft", "LLaMA 2",    "Llama-2-13B"),
    "WizardLM-70B-V1.0":  ("Microsoft", "LLaMA 2",    "Llama-2-70B"),
    "WizardCoder-15B":    ("Microsoft", "StarCoder",   "StarCoder (15.5B)"),
    "WizardMath-7B-V1.0": ("Microsoft", "LLaMA 2",    "Llama-2-7B"),
    "WizardMath-7B-V1.1": ("Microsoft", "Mistral",     "Mistral v0.1 (7B)"),
    "WizardMath-13B-V1.0":("Microsoft", "LLaMA 2",    "Llama-2-13B"),
    "WizardMath-70B-V1.0":("Microsoft", "LLaMA 2",    "Llama-2-70B"),
    "Orca-Math 7B":        ("Microsoft", "Mistral",     "Mistral v0.1 (7B)"),
    "LRV-Instruct":        ("Microsoft", "LLaVA",      "LLaMA 13B"),

    # ── Abacus.AI ──────────────────────────────────────────────────────────────
    # (developer upgraded from slug "abacusai")
    "Dracarys-72B-Instruct": ("Abacus.AI", "Qwen", "Qwen2.5-72B"),

    # ── LMSYS ──────────────────────────────────────────────────────────────────
    # (developer upgraded from slug "lmsys")
    "longchat-13b-16k":    ("LMSYS", "LLaMA",   "LLaMA 13B"),
    "longchat-7b-v1.5-32k":("LMSYS", "LLaMA 2", "Llama-2-7B"),

    # ── OpenChat ───────────────────────────────────────────────────────────────
    # (developer upgraded from slug "openchat")
    "openchat_v2":              ("OpenChat", "LLaMA",    "LLaMA 13B"),
    "openchat_v2_w":            ("OpenChat", "LLaMA",    "LLaMA 13B"),
    "openchat_8192":            ("OpenChat", "LLaMA",    "LLaMA 13B"),
    "openchat_v3.1":            ("OpenChat", "LLaMA 2",  "Llama-2-13B"),
    "openchat_v3.2":            ("OpenChat", "LLaMA 2",  "Llama-2-13B"),
    "openchat_v3.2_super":      ("OpenChat", "LLaMA 2",  "Llama-2-13B"),
    "openchat_3.5":             ("OpenChat", "Mistral",  "Mistral v0.1 (7B)"),
    "openchat-3.5-0106":        ("OpenChat", "Mistral",  "Mistral v0.1 (7B)"),
    "openchat-3.5-1210":        ("OpenChat", "Mistral",  "Mistral v0.1 (7B)"),
    "openchat-3.6-8b-20240522": ("OpenChat", "LLaMA 3",  "Llama 3 (8B)"),
    "opencoderplus":            ("OpenChat", "StarCoder","StarCoder (15.5B)"),
    "OpenChat-3.5 7B":          ("OpenChat", "Mistral",  "Mistral v0.1 (7B)"),
    "OpenChat-3.5-1210 7B":     ("OpenChat", "Mistral",  "Mistral v0.1 (7B)"),

    # ── Speakleash (Bielik — Polish LLM trained from scratch on Mistral arch) ──
    # (developer upgraded from slug "speakleash")
    "Bielik-11B-v2.0-Instruct": ("Speakleash", "Bielik", "Bielik-11B-v2"),
    "Bielik-11B-v2.1-Instruct": ("Speakleash", "Bielik", "Bielik-11B-v2"),
    "Bielik-11B-v2.2-Instruct": ("Speakleash", "Bielik", "Bielik-11B-v2"),
    "Bielik-11B-v2.3-Instruct": ("Speakleash", "Bielik", "Bielik-11B-v2"),

    # ── Teknium ────────────────────────────────────────────────────────────────
    # (developer upgraded from slug "teknium")
    "OpenHermes-13B": ("Teknium", "LLaMA", "LLaMA 13B"),
    "OpenHermes-7B":  ("Teknium", "LLaMA", "LLaMA 7B"),

    # ── AI21 Labs ──────────────────────────────────────────────────────────────
    "Jamba Instruct": ("AI21 Labs", "Jamba", "Jamba"),

    # ── Upstage (Solar Pro — Upstage's own 22B SOLAR-arch model) ────────────
    "Solar Pro": ("Upstage", "SOLAR", ""),

    # ── Marin Community (Stanford/Together AI research) ────────────────────────
    "Marin 8B Instruct": ("Marin Community", "Marin", "Marin-8B"),

    # ── Sea AI Lab (Sailor — Qwen2 continual pretraining for SEA langs) ────────
    "Sailor Chat (14B)": ("Sea AI Lab", "Sailor", "Qwen2 14B"),
    "Sailor Chat (7B)":  ("Sea AI Lab", "Sailor", "Qwen2 7B"),

    # ── SambaNova Systems ──────────────────────────────────────────────────────
    "SambaLingo-Thai-Chat":     ("SambaNova Systems", "LLaMA 2", "Llama-2-7B"),
    "SambaLingo-Thai-Chat-70B": ("SambaNova Systems", "LLaMA 2", "Llama-2-70B"),

    # ── AI Singapore (SEA-LION — MPT-based SEA multilingual model) ────────────
    "SEA-LION 7B Instruct": ("AI Singapore", "SEA-LION", "SEA-LION v1 (7B)"),

    # ── FreedomIntelligence / HKUST (AceGPT v2 — Arabic/multilingual) ─────────
    "AceGPT-v2-8B-Chat":  ("FreedomIntelligence", "AceGPT", "Llama 3 (8B)"),
    "AceGPT-v2-32B-Chat": ("FreedomIntelligence", "AceGPT", "Qwen2.5-32B"),
    "AceGPT-v2-70B-Chat": ("FreedomIntelligence", "AceGPT", "Llama 3.1 (70B)"),

    # ── ALLaM (SDAIA / KACST — Arabic LLM) ────────────────────────────────────
    "ALLaM-7B-Instruct-preview": ("ALLaM", "ALLaM", "LLaMA (7B)"),

    # ── Typhoon AI / SCB 10X (Thai LLM on LLaMA 3) ────────────────────────────
    "Typhoon v1.5 Instruct (8B)":  ("Typhoon AI", "LLaMA 3", "Llama 3 (8B)"),
    "Typhoon 1.5X instruct (8B)":  ("Typhoon AI", "LLaMA 3", "Llama 3 (8B)"),

    # ── Salesforce (InstructBLIP — BLIP-2 + instruction tuning) ───────────────
    "InstructBLIP":     ("Salesforce", "BLIP", "BLIP-2"),
    "InstructBLIP-7B":  ("Salesforce", "BLIP", "BLIP-2"),
    "InstructBLIP-13B": ("Salesforce", "BLIP", "BLIP-2"),

    # ── Shanghai AI Lab (VideoChat — BLIP-2 visual + Vicuna LLM) ──────────────
    "VideoChat": ("Shanghai AI Lab", "VideoChat", "BLIP-2"),

    # ── MBZUAI (Video-ChatGPT — LLaVA framework + LLaMA 13B) ─────────────────
    "Video-ChatGPT": ("MBZUAI", "LLaVA", "LLaMA 13B"),

    # ── PKU YuanGroup (MovieChat — Q-Former + Vicuna for long video) ──────────
    "MovieChat":  ("PKU YuanGroup", "MovieChat", "Vicuna-7B"),
    "MovieChat+": ("PKU YuanGroup", "MovieChat", "Vicuna-7B"),

    # ── BAAI (Emu2 — LLaMA 2 33B multimodal model) ────────────────────────────
    "Emu2-Chat": ("BAAI", "Emu", "LLaMA 2 (33B)"),

    # ── Zhipu AI / Tsinghua (CogVLM — ViT + Vicuna 7B) ───────────────────────
    "CogVLM-Chat": ("Zhipu AI", "CogVLM", "Vicuna-7B"),

    # ── Huawei (PanGu-Coder fine-tune) ────────────────────────────────────────
    "PanGu-Coder-FT-I": ("Huawei", "PanGu", "PanGu-Coder"),

    # ── FreedomIntelligence (Phoenix — BLOOMZ-based multilingual chat) ─────────
    "phoenix-inst-chat-7b": ("FreedomIntelligence", "BLOOMZ", "BLOOMZ (7.1B)"),

    # ── GPT-Neo fine-tunes (Bhāskara / Neo, Indic language models) ────────────
    "Bhāskara-P (Fine-tuned, 2.7B)": ("", "GPT-Neo", "GPT-Neo (2.7 B)"),
    "Bhāskara-A (Fine-tuned, 2.7B)": ("", "GPT-Neo", "GPT-Neo (2.7 B)"),
    "Neo-P (Fine-tuned, 2.7B)":       ("", "GPT-Neo", "GPT-Neo (2.7 B)"),
    "Neo-A (Fine-tuned, 2.7B)":       ("", "GPT-Neo", "GPT-Neo (2.7 B)"),

    # ── Apple (MM1 — multimodal; chat variants fine-tuned from base) ──────────
    "MM1-3B-Chat":  ("Apple", "MM1", "MM1-3B"),
    "MM1-7B-Chat":  ("Apple", "MM1", "MM1-7B"),
    "MM1-30B-Chat": ("Apple", "MM1", "MM1-30B"),

    # ── LLaVA-AlignedVQ (LLaVA variant with VQ alignment) ─────────────────────
    "LLaVA-AlignedVQ": ("", "LLaVA", "LLaVA"),

    # ── Infinigence AI ─────────────────────────────────────────────────────────
    "InfMLLM-7B-Chat": ("Infinigence AI", "", ""),

    # ── Unknown / not findable on HF ───────────────────────────────────────────
    "MediSwift-XL":  ("", "", ""),
    "MiCo-Chat-7B":  ("", "", ""),
}


def apply(m_df, write: bool):
    m = m_df.copy()
    updated = 0

    for model_id, (dev, fam, base) in ENRICH.items():
        mask = m["model_id"] == model_id
        if not mask.any():
            print(f"  WARN  not found in models.csv: {model_id!r}")
            continue

        changes = []
        # developer: always write if non-blank (upgrading slugs to display names)
        if dev and m.loc[mask, "developer"].values[0] != dev:
            old = m.loc[mask, "developer"].values[0]
            changes.append(f"developer: {old!r} → {dev!r}")
            if write:
                m.loc[mask, "developer"] = dev
        # model_family: only fill blank
        if fam and not m.loc[mask, "model_family"].values[0]:
            changes.append(f"model_family: '' → {fam!r}")
            if write:
                m.loc[mask, "model_family"] = fam
        # base_model: only fill blank
        if base and not m.loc[mask, "base_model"].values[0]:
            changes.append(f"base_model: '' → {base!r}")
            if write:
                m.loc[mask, "base_model"] = base

        if changes:
            print(f"  {model_id}")
            for c in changes:
                print(f"    {c}")
            updated += 1

    return m, updated


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    print("Loading files...")
    _, m_df, _ = io.load_data()
    print(f"  models: {len(m_df)}")

    print(f"\n{'Applying' if args.write else 'Dry run —'} metadata enrichment:\n")
    m_new, n_updated = apply(m_df, write=args.write)

    if args.write:
        io.save_csv(m_new, config.MODELS_CSV)
        print(f"\nWritten. {n_updated} models updated.")
        print("\nNext step:")
        print("  python3 scripts/verify_data.py")
    else:
        print(f"\nDry run summary: {n_updated} models would be updated.")
        print("Pass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
