#!/usr/bin/env python3
"""
Compound / Prompting-Method Cleanup — Pass 5
=============================================
Three kinds of operations following the final model-family audit:

  REMOVE     — cascade-delete compound/pipeline models and ambiguous prompting
               method entries with no recoverable base model
  RENAME     — rename base-less entries in-place (also patches results.setup)
  REMAP      — merge into existing canonical model
  SETUP_REMAP— extract setup qualifier into results.setup, merge into base model

Compound models (multiple separate models/tools used as a pipeline) are
excluded from the dataset per methodology.  Prompting methods that encode the
base model in the name have setup separated and are merged into that base.

Usage:
  python3 scripts/standardise_pass5.py           # dry run
  python3 scripts/standardise_pass5.py --write   # apply

Always run afterwards:
  python3 scripts/verify_data.py
  python3 scripts/manage_data.py recompute-stats --write
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib import io, config

# ─────────────────────────────────────────────────────────────────────────────
# REMOVE — cascade-delete model row + all result rows
# Reasons: pipeline/compound system; prompting-method label without base model
# ─────────────────────────────────────────────────────────────────────────────
REMOVE = {
    # ── Transcription-pipeline + GPT-4o (HELM Audio) ─────────────────────────
    "GPT-4o Transcribe + GPT-4o (2024-11-20)":        "compound: ASR Transcribe API + chat API",
    "GPT-4o mini Transcribe + GPT-4o (2024-11-20)":   "compound: ASR Transcribe API + chat API",
    "Whisper-1 + GPT-4o (2024-11-20)":                "compound: Whisper ASR + GPT-4o chat API",

    # ── DePlot compound systems ───────────────────────────────────────────────
    "DePlot + GPT-4":                               "compound: DePlot chart parser + GPT-4 LLM",
    "DePlot+GPT3":                                  "compound: DePlot chart parser + GPT-3",
    "DePlot+FlanPaLM":                              "compound: DePlot + FlanPaLM",
    "DePlot+Codex (PoT Self-Consistency)":          "compound: DePlot + Codex PoT",
    "DePlot+FlanPaLM+Codex (PoT Self-Consistency)":"compound: DePlot + FlanPaLM + Codex",
    "DePlot+GPT3 (Self-Consistency)":              "compound: DePlot + GPT-3 (already handled base)",
    "DePlot+FlanPaLM (Self-Consistency)":           "compound: DePlot + FlanPaLM (already handled)",
    "StructChart+GPT3.5 (STR)":                     "compound: StructChart + GPT-3.5",
    "StructChart+GPT3.5 (STR ChartQA+SimChart9K)":  "compound: StructChart + GPT-3.5",

    # ── Code-generation agentic pipelines ────────────────────────────────────
    "LPW (GPT-4o)":                       "compound: LLM-Planner-Writer pipeline using GPT-4o",
    "MapCoder (GPT-4)":                   "compound: MapCoder multi-agent system using GPT-4",
    "MapCoder (GPT-4o)":                  "compound: MapCoder multi-agent system using GPT-4o",
    "L2MAC (GPT-4)":                      "compound: L2MAC code generation agent using GPT-4",
    "AFlow(GPT-4o-mini)":                 "compound: AFlow agentic workflow using GPT-4o-mini",
    "CodeSim (GPT4)":                     "compound: CodeSim simulation pipeline using GPT-4",
    "CodeSim (GPT4o)":                    "compound: CodeSim simulation pipeline using GPT-4o",
    "GPT-3.5 Turbo (Self-Debugging with unit tests + trace)": "compound: self-debugging scaffold",
    "GPT-4 (Self-Debugging with unit tests + trace)":         "compound: self-debugging scaffold",
    "StarCoder 15.5B (Self-Debugging with unit tests + trace)":"compound: self-debugging scaffold",
    "code-davinci-002 175B (Self-Debugging with unit tests + trace)": "compound: self-debugging scaffold",
    "GPT-3.5 Turbo + INTERVENOR":         "compound: INTERVENOR iterative code repair pipeline",
    "GPT-3.5 Turbo + Language Agent Tree Search": "compound: LATS tree-search wrapper",
    "GPT-3.5 Turbo (ChatGPT) + AgentCoder": "compound: AgentCoder pipeline",
    "GPT-4 + AgentCoder":                 "compound: AgentCoder pipeline",

    # ── Retrieval-augmented / web-search pipelines ────────────────────────────
    "GPT-4 + WebSearch":                  "compound: GPT-4 with web search tool",
    "GPT-4 + knowledge base":             "compound: GPT-4 with knowledge base retrieval",
    "RAPTOR + GPT-4 (June 2023)":         "compound: RAPTOR RAG system + GPT-4",
    "RALM (LLaMA2-13B + Google Search)":  "compound: Retrieval-Augmented LM with search",
    "MCR (code-davinci-002) + Google Search": "compound: MCR reasoning + Google Search",
    "Self-ask (GPT-3; davinci-002) + Google Search": "compound: Self-Ask + web search",
    "ReST meets ReAct (PaLM 2-L + Google Search)": "compound: ReAct agent + Google Search",
    "Rethinking with retrieval (GPT-3)":  "compound: retrieval-augmented prompting pipeline",

    # ── Visual-language compound systems ─────────────────────────────────────
    "BLIP-2 + ChatGPT":                   "compound: BLIP-2 visual encoder + ChatGPT LLM",
    "ChatCaptioner + ChatGPT":            "compound: ChatCaptioner captioning + ChatGPT",
    "InstructBLIP + ChatGPT + Neuro-Symbolic": "compound: InstructBLIP + ChatGPT + NS module",
    "InstructBLIP + GPT-4":               "compound: InstructBLIP + GPT-4 LLM",
    "IVM-Enhanced GPT4-V":                "compound: IVM visual prompt + GPT-4V",
    "ChartPaLI-5B + PaLM 2-S":            "compound: ChartPaLI chart model + PaLM 2-S LLM",
    "CLIP + PaLM (540B)":                  "compound: CLIP visual encoder + PaLM",
    "CLIP + FiD":                          "compound: CLIP retrieval + Fusion-in-Decoder",
    "GaC(Qwen2-72B-Instruct + Llama-3-70B-Instruct)": "compound: two-LLM Gen and Check pipeline",

    # ── Prompting-framework pipelines ────────────────────────────────────────
    "MMCTAgent (GPT-4 + GPT-4V)":         "compound: multi-modal agent using GPT-4 + GPT-4V",
    "SKiC (GPT-4 model)":                  "compound: Skills-in-Context prompting framework",
    "PHP (GPT-4 model)":                   "compound: Progressive Hint Prompting framework",
    "KeyComp (GPT-3.5)":                   "compound: KeyComp compositional prompting pipeline",
    "KeyComp* (GPT-3.5)":                  "compound: KeyComp* variant",
    "KeyComp* (GPT-4)":                    "compound: KeyComp* variant using GPT-4",
    "OCaTS (kNN & GPT-3.5-turbo":          "compound: kNN retrieval + GPT-3.5 pipeline",
    "OpenPipe-MoA-GPT4-Turbo":             "compound: Mixture-of-Agents pipeline",
    "o1-mini + Language Agent Tree Search (Hamming.ai)": "compound: LATS over o1-mini",
    "o1-mini + MapCoder (Hamming.ai)":     "compound: MapCoder pipeline over o1-mini",
    "DUP prompt upon GPT-4":               "compound: Diversified Unified Prompting framework",
    "Finetuned GPT-3 175B + verifier":     "compound: GPT-3 + separate verifier model",
    "CR (GPT-4 model, w/o code)":          "compound: Code Reasoning system using GPT-4",
    "CR (GPT-4-turbo model, w/ code)":     "compound: Code Reasoning system using GPT-4-Turbo",
    "GPT-4-code model (w/ code)":          "compound: GPT-4 code execution pipeline",
    "GPT-4-code model (w/o code)":         "compound: GPT-4 code execution pipeline (no exec)",
    "GPT-4-code model (CSV, w/ code)":     "compound: GPT-4 code pipeline with CSV",
    "GPT-4-code model (CSV, w/ code, SC, k=16)": "compound: GPT-4 code pipeline + SC",

    # ── QurrentOS coding pipeline ─────────────────────────────────────────────
    "QurrentOS-coder + GPT-4":             "compound: QurrentOS orchestration + GPT-4",
    "QurrentOS-coder + GPT-4 Turbo":       "compound: QurrentOS orchestration + GPT-4 Turbo",
    "QurrentOS-coder + GPT-4o":            "compound: QurrentOS orchestration + GPT-4o",
    "QurrentOS-coder + Llama 3 70b":       "compound: QurrentOS orchestration + Llama-3 70B",
    "QurrentOS-coder + Qwen-72B-Instruct": "compound: QurrentOS orchestration + Qwen-72B",
    "QurrentOS-coder + Claude 3 Opus":     "compound: QurrentOS orchestration + Claude 3 Opus",
    "QurrentOS-coder + Claude 3.5 Sonnet": "compound: QurrentOS orchestration + Claude 3.5 Sonnet",
    "QurrentOS-coder + DeepSeek-Coder-V2": "compound: QurrentOS orchestration + DeepSeek-Coder-V2",
    "QurrentOS-coder + Gemini 1.5 Pro":    "compound: QurrentOS orchestration + Gemini 1.5 Pro",

    # ── Compound Phi-GSM ─────────────────────────────────────────────────────
    "Phi-GSM+V 1.3B+1.3B (verify48@1)":   "compound: Phi-GSM (1.3B) + Phi-V verifier (1.3B)",

    # ── Prompting methods with no recoverable base model ─────────────────────
    "Auto-CoT":                  "prompting method paper; base model not recorded in entry",
    "CoT":                       "generic CoT label; base model not recoverable",
    "MC-CoT":                    "generic MC-CoT label; base model not recoverable",
    "CoT_Eng (self-consistency @ 5)": "prompting method; base model not recoverable",
    "PoT_Eng (self-consistency @ 5)": "prompting method; base model not recoverable",
    "Self-Ask":                  "prompting method; base model not recorded",

    # ── Gemini + CoT (Gemini base version ambiguous) ─────────────────────────
    "Gemini + CCoT":  "compound: Gemini base version ambiguous for this winoground entry",
    "Gemini + CoCoT": "compound: Gemini base version ambiguous for this winoground entry",
    "Gemini + DDCoT": "compound: Gemini base version ambiguous for this winoground entry",
    "Gemini-2.0 + CA":"compound: Gemini 2.0 version ambiguous; CA = Compositional Augmentation",

    # ── MMICL (base model not in dataset) ─────────────────────────────────────
    "MMICL + CCoT":   "compound: MMICL base model not in dataset",
    "MMICL + CoCoT":  "compound: MMICL base model not in dataset",
    "MMICL + DDCoT":  "compound: MMICL base model not in dataset",

    # ── Aggregate placeholder ─────────────────────────────────────────────────
    "Gemini 1.5 Family": "aggregate placeholder, not a specific model",
}

# ─────────────────────────────────────────────────────────────────────────────
# RENAME — no existing canonical; update model entry in-place.
# RENAME_SETUP maps new_id → setup_val to set on results after renaming.
# Applied BEFORE REMAP and SETUP_REMAP.
# ─────────────────────────────────────────────────────────────────────────────
RENAME = {
    # ToRA base entries (no canonical without "w/ code" qualifier)
    "ToRA 7B (w/ code)":   "ToRA 7B",    # 1r
    "ToRA 13B (w/ code)":  "ToRA 13B",   # 1r
    # OVM verifier models (fine-tuned; base doesn't exist yet)
    "OVM-Mistral-7B (verify100@1)": "OVM-Mistral-7B",  # 1r — verify@100 is setup
    "OVM-Llama2-7B (verify100@1)":  "OVM-Llama2-7B",   # 1r — verify@100 is setup
    # LaMDA: base doesn't exist; prompting-method label becomes base
    "Few-shot CoT LaMDA 137B": "LaMDA 137B",   # 1r — setup="Few-shot CoT"
    # MiniGPT-4 casing fix
    "MiniGPT4-13B": "MiniGPT-4-13B",  # 1r
}

# Setup values to apply to renamed entries' result rows
RENAME_SETUP = {
    "ToRA 7B":        "w/ code",
    "ToRA 13B":       "w/ code",
    "OVM-Mistral-7B": "verify@100",
    "OVM-Llama2-7B":  "verify@100",
    "LaMDA 137B":     "Few-shot CoT",
}

# ─────────────────────────────────────────────────────────────────────────────
# REMAP — merge into existing canonical; delete old entry
# ─────────────────────────────────────────────────────────────────────────────
REMAP = {
    # miniGPT4 → MiniGPT-4-7B (same model, different capitalisation)
    "miniGPT4":    "MiniGPT-4-7B",    # 1r
    # GPT4RoI → GPT4ROI 7B (ROI) (same 7B model, more descriptive name exists)
    "GPT4RoI":     "GPT4ROI 7B (ROI)",# 2r
    # GPT-3.5 Turbo (175B) / (ChatGPT) → GPT-3.5-Turbo (parameter count redundant)
    "GPT-3.5 Turbo (175B)":   "GPT-3.5-Turbo",  # 2r
    "GPT-3.5 Turbo (ChatGPT)":"GPT-3.5-Turbo",  # 1r
}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP_REMAP — old_id → (base_id, setup_val)
# Moves setup qualifier into results.setup, then merges into base model.
# ─────────────────────────────────────────────────────────────────────────────
SETUP_REMAP = {
    # ── CoT/ZS-CoT applied to existing multimodal models ─────────────────────
    "InstructBLIP-ZS-CoT":    ("InstructBLIP", "ZS-CoT"),   # 3r
    "LLaVA-1.5-ZS-CoT":       ("LLaVA-1.5",   "ZS-CoT"),   # 3r
    "GPT-4V + CoCoT":          ("GPT-4V",       "CoCoT"),    # 3r
    "OpenFlamingo + CoCoT":    ("OpenFlamingo", "CoCoT"),    # 3r

    # ── Chain-of-Thought / prompting setups on GPT-3 ─────────────────────────
    "Chain-of-Thought (GPT-3; davinci-002)": ("GPT-3 175B", "CoT"),             # 1r
    "Direct Prompting (GPT-3; davinci-002)": ("GPT-3 175B", "Direct Prompting"),# 1r
    "Self-ask (GPT-3; davinci-002)":         ("GPT-3 175B", "Self-Ask"),         # 1r

    # ── Few-shot CoT on specific models ──────────────────────────────────────
    "Few-shot CoT GPT-J":           ("GPT-J (6B)",      "Few-shot CoT"),  # 1r
    "Few-shot gpt-3.5-turbo CoT":   ("GPT-3.5-Turbo",   "Few-shot CoT"),  # 1r

    # ── Prompting augmentation on GPT-4V / GPT-4o ────────────────────────────
    "GPT4-Vision 4-shot": ("GPT-4V",  "4-shot"),  # 1r  (4-shot is evaluation setup)
    "GPT-4o + CA":        ("GPT-4o",  "CA"),       # 4r  (CA = Compositional Augmentation)

    # ── Selected-Demo & Uncertainty prompting on multiple models ─────────────
    "GPT-3.5-Turbo w/ Selected Demo & Uncertainty": ("GPT-3.5-Turbo",  "Selected Demo & Uncertainty"), # 2r
    "LLaMA-2-13B w/ Selected Demo & Uncertainty":   ("Llama 2 (13B)", "Selected Demo & Uncertainty"), # 2r
    "LLaMA-2-70B w/ Selected Demo & Uncertainty":   ("Llama 2 (70B)", "Selected Demo & Uncertainty"), # 2r

    # ── SigExt summarization prompting ───────────────────────────────────────
    "Claude Instant + SigExt":  ("Claude Instant 1.2",  "SigExt"),  # 4r
    "Mistral 7B + SigExt":      ("Mistral v0.1 (7B)",   "SigExt"),  # 2r

    # ── ToRA (w/ code) → existing base entries ────────────────────────────────
    "ToRA 70B (w/ code)":     ("ToRA 70B",      "w/ code"),  # 1r
    "ToRA-Code 7B (w/ code)": ("ToRA-Code 7B",  "w/ code"),  # 1r
    "ToRA-Code 13B (w/ code)":("ToRA-Code 13B", "w/ code"),  # 1r
    "ToRA-Code 34B (w/ code)":("ToRA-Code 34B", "w/ code"),  # 1r

    # ── OVM verify@20 → OVM-Mistral-7B (created by RENAME above) ─────────────
    "OVM-Mistral-7B (verify20@1)": ("OVM-Mistral-7B", "verify@20"),  # 1r
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
    for mid in REMOVE:
        if mid not in existing:
            print(f"  WARN  REMOVE target missing: {mid!r}")
    return ok


def apply(m_df, r_df, write: bool):
    m = m_df.copy()
    r = r_df.copy()

    # ── 1. REMOVE ─────────────────────────────────────────────────────────────
    total_removed_r = 0
    for mid, reason in REMOVE.items():
        cur = set(m["model_id"]) if write else set(m_df["model_id"])
        if mid not in cur:
            print(f"  WARN  REMOVE missing: {mid!r}")
            continue
        n = (r_df["model_name"] == mid).sum()
        print(f"  REMOVE  {mid!r}  ({n}r)")
        total_removed_r += n
        if write:
            m = m[m["model_id"] != mid]
            r = r[r["model_name"] != mid]

    # ── 2. RENAME ─────────────────────────────────────────────────────────────
    for old_id, new_id in RENAME.items():
        cur = set(m["model_id"]) if write else set(m_df["model_id"])
        if old_id not in cur:
            print(f"  WARN  RENAME source missing: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  RENAME  {old_id!r} -> {new_id!r}  ({n}r)")
        if write:
            m.loc[m["model_id"]   == old_id, "model_id"]   = new_id
            m.loc[m["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_id"]   == old_id, "model_id"]   = new_id

    # Patch setup for renamed entries
    if write:
        for new_id, setup_val in RENAME_SETUP.items():
            mask = (r["model_name"] == new_id) & (r["setup"] == "")
            r.loc[mask, "setup"] = setup_val

    # ── 3. REMAP ──────────────────────────────────────────────────────────────
    for old_id, new_id in REMAP.items():
        cur = set(m["model_id"]) if write else set(m_df["model_id"])
        if old_id not in cur:
            print(f"  WARN  REMAP source missing: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  REMAP  {old_id!r} -> {new_id!r}  ({n}r)")
        if write:
            r.loc[r["model_name"] == old_id, "model_name"] = new_id
            r.loc[r["model_id"]   == old_id, "model_id"]   = new_id
            m = m[m["model_id"] != old_id]

    # ── 4. SETUP_REMAP ────────────────────────────────────────────────────────
    for old_id, (base_id, setup_val) in SETUP_REMAP.items():
        cur = set(m["model_id"]) if write else (set(m_df["model_id"]) | set(RENAME.values()))
        if old_id not in cur:
            print(f"  WARN  SETUP_REMAP source missing: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  SETUP_REMAP  {old_id!r} -> {base_id!r}  setup={setup_val!r}  ({n}r)")
        if write:
            mask = r["model_name"] == old_id
            r.loc[mask & (r["setup"] == ""), "setup"] = setup_val
            r.loc[mask, "model_name"] = base_id
            r.loc[r["model_id"] == old_id, "model_id"] = base_id
            m = m[m["model_id"] != old_id]

    # ── 5. Post-merge dedup ───────────────────────────────────────────────────
    dupes_dropped = 0
    if write:
        key_cols = [c for c in config.RESULT_IDENTITY_KEY if c in r.columns]
        before = len(r)
        r = r.drop_duplicates(subset=key_cols).reset_index(drop=True)
        dupes_dropped = before - len(r)

    return m, r, total_removed_r, dupes_dropped


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
    m_new, r_new, n_removed_r, n_dupes = apply(m_df, r_df, write=args.write)

    if args.write:
        io.save_csv(m_new, config.MODELS_CSV)
        io.save_results(r_new)
        delta_m = len(m_df) - len(m_new)
        delta_r = len(r_df) - len(r_new)
        print(f"\nWritten.")
        print(f"  models:  {len(m_df)} → {len(m_new)}  (−{delta_m})")
        print(f"  results: {len(r_df)} → {len(r_new)}  "
              f"(−{delta_r}, {n_dupes} post-merge dupes dropped)")
        print(f"  result rows directly removed: {n_removed_r}")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        print(f"\nDry run summary:")
        print(f"  REMOVE:      {len(REMOVE)} models")
        print(f"  RENAME:      {len(RENAME)}")
        print(f"  REMAP:       {len(REMAP)}")
        print(f"  SETUP_REMAP: {len(SETUP_REMAP)}")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
