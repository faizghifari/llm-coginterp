#!/usr/bin/env python3
"""
Model Family Standardisation Pass 2
=====================================
Follows the setup-extraction pass (fix_setup_in_names.py) to handle patterns
that the first regex missed, plus cross-family naming duplicates discovered
during the full family audit.

Operations:
  REMOVE  — cascade-delete model + all its result rows
  REMAP   — redirect results to existing canonical, delete old model row
  RENAME  — change model_id in-place (in models.csv AND results.csv)
  SETUP   — like REMAP but also sets results.setup / results.reasoning_enabled
             when the setup/reasoning info is encoded in the model name

Missed setup patterns now covered:
  • (few-shot, k=N)   →  setup column
  • (N-shot, ...)      →  setup column
  • (zero-shot, ...)   →  setup column
  • (thinking)         →  reasoning_enabled = True
  • (extended thinking) in a comma-separated suffix → reasoning_enabled = True
  • (thinking disabled) → setup = "thinking disabled"

Usage:
  python3 scripts/standardise_models.py           # dry run
  python3 scripts/standardise_models.py --write   # apply
"""
import argparse, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scripts.lib import io, config

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_setup(raw: str) -> str:
    """Normalize extracted setup string to a canonical display value."""
    s = raw.strip()
    # Normalise shot counts: "few-shot, k=5" → "5-shot"
    m = re.match(r'few[\-\s]shot,?\s*k=(\d+)', s, re.IGNORECASE)
    if m: return f'{m.group(1)}-shot'
    m = re.match(r'(\d+)[\-\s]shot', s, re.IGNORECASE)
    if m: return f'{m.group(1)}-shot'
    m = re.match(r'zero[\-\s]shot', s, re.IGNORECASE)
    if m: return 'Zero-shot'
    m = re.match(r'0[\-\s]shot', s, re.IGNORECASE)
    if m: return '0-shot'
    # Prefix-normalise zero/0-shot with extra content
    s2 = re.sub(r'^zero[\-\s]shot', 'Zero-shot', s, flags=re.IGNORECASE)
    s2 = re.sub(r'^0[\-\s]shot', '0-shot', s2, flags=re.IGNORECASE)
    s2 = re.sub(r'^few[\-\s]shot,?\s*k=(\d+)', lambda m: f'{m.group(1)}-shot', s2, flags=re.IGNORECASE)
    return s2.strip(', ')


# Regex for patterns the first pass missed.
# Captures: base_group(1), setup_group(2)
# The trailing parenthetical may contain:
#   few-shot/N-shot/zero-shot with additional , k=N or , detail text
#   thinking / extended thinking / thinking disabled
MISSED_RE = re.compile(
    r'^(.*?)\s*\(('
    r'few[\-\s]shot.*'          # few-shot, k=N  etc.
    r'|\d+[\-\s]shot.*'         # N-shot, something
    r'|zero[\-\s]shot,.+'       # zero-shot, with extra detail
    r'|0[\-\s]shot,.+'          # 0-shot, with extra detail
    r'|extended thinking'       # extended thinking (possibly with date prefix below)
    r'|thinking disabled'
    r'|thinking'
    r')\)\s*$',
    re.IGNORECASE
)

# Special: "Claude 4 Opus (20250514, extended thinking)" — date+thinking combined
DATE_THINK_RE = re.compile(
    r'^(.*?)\s*\(\d{8},\s*extended thinking\)\s*$',
    re.IGNORECASE
)


def detect_missed_setups(model_ids: set) -> list[tuple[str, str, str, bool, bool]]:
    """
    Returns list of (old_id, base_id, setup_val, is_thinking, is_thinking_disabled).
    is_thinking=True → set reasoning_enabled=True in results (don't write to setup col).
    is_thinking_disabled=True → write "thinking disabled" to setup col.
    """
    entries = []
    for mid in sorted(model_ids):
        # Special case: date + extended thinking suffix
        m = DATE_THINK_RE.match(mid)
        if m:
            base = m.group(1).strip()
            entries.append((mid, base, '', True, False))
            continue
        m = MISSED_RE.search(mid)
        if not m:
            continue
        base = m.group(1).strip()
        raw_setup = m.group(2).strip()
        low = raw_setup.lower()
        is_think   = low in ('thinking',)
        is_nodthink = low == 'thinking disabled'
        setup_val  = '' if is_think else ('thinking disabled' if is_nodthink else _norm_setup(raw_setup))
        entries.append((mid, base, setup_val, is_think, is_nodthink))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE REMOVE
# ─────────────────────────────────────────────────────────────────────────────
REMOVE = {
    "Claude-3 Family": "aggregate placeholder, not a specific model",
}

# ─────────────────────────────────────────────────────────────────────────────
# RENAME  (model row updated in-place; results model_name/model_id updated)
# ─────────────────────────────────────────────────────────────────────────────
RENAME = {
    # Claude
    "claude_instant_1.2":        "Claude Instant 1.2",
    # DeepSeek
    "deepseek-r1-zero":          "DeepSeek-R1-Zero",  # distinct from R1 (no SFT)
    # Mistral
    "MiniStral 3Mistral AI":     "Ministral 3B",
    # Gemini — strip HELM-style hyphenated names to proper spacing
    "Gemini-1.5-Pro":            "Gemini 1.5 Pro",
    "Gemini-2.5-Flash":          "Gemini 2.5 Flash",
    "Gemini-2.5-Flash-Lite":     "Gemini 2.5 Flash-Lite",
    "Gemini-V":                  "Gemini Pro Vision",   # Gemini Vision = Gemini Pro Vision
}

# ─────────────────────────────────────────────────────────────────────────────
# REMAP  (old_id → existing canonical; results repointed; old model row deleted)
# All targets verified to exist in models.csv before applying.
# ─────────────────────────────────────────────────────────────────────────────
REMAP = {
    # ── BLOOM ─────────────────────────────────────────────────────────────────
    "BLOOM-176B":                        "BLOOM 176B",

    # ── Claude ────────────────────────────────────────────────────────────────
    "Claude v1.3":                       "Claude 1.3",
    "Claude-1":                          "Claude 1.3",     # 1-result, June 2024 paper → v1.3
    "Claude-2":                          "Claude 2",
    "Claude-Opus":                       "Claude 3 Opus",  # 1-result, June 2024 paper
    "Claude 3 Haiku (20240307)":         "Claude 3 Haiku",
    "Claude 3 Opus (20240229)":          "Claude 3 Opus",
    "Claude 3 Sonnet (20240229)":        "Claude 3 Sonnet",
    # Claude 3.5 Sonnet consolidation (target created below via RENAME chain)
    "Claude Sonnet 3.5":                 "Claude 3.5 Sonnet",
    "Claude-3.5-Sonnet":                 "Claude 3.5 Sonnet",
    "Claude3.5-Sonnet":                  "Claude 3.5 Sonnet",
    # Claude 4 naming: "Claude 4 Opus/Sonnet (date)" → canonical "Claude Opus/Sonnet 4"
    "Claude 4 Opus (20250514)":          "Claude Opus 4",
    "Claude 4 Sonnet (20250514)":        "Claude Sonnet 4",
    # Claude 4.x date-version → canonical without date + different word order
    "Claude 4.5 Haiku (20251001)":       "Claude Haiku 4.5",
    "Claude 4.5 Sonnet (20250929)":      "Claude Sonnet 4.5",
    "Claude 4.6 Sonnet":                 "Claude Sonnet 4.6",
    "Claude 4.7 Opus":                   "Claude Opus 4.7",

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    "DeepSeek-reasoner":                 "DeepSeek-R1",  # "reasoner" was the API alias

    # ── Gemini ────────────────────────────────────────────────────────────────
    "Google/Gemini 2.5 Pro":             "Gemini 2.5 Pro",

    # ── GPT / Vision variants ─────────────────────────────────────────────────
    "GPT4-Vision":                       "GPT-4V",
    "GPT4V":                             "GPT-4V",

    # ── SparseGPT format normalisation ───────────────────────────────────────
    "SparseGPT (175B, 2:4 Sparsity)":   "SparseGPT 175B (2:4 Sparsity)",
    "SparseGPT (175B, 4:8 Sparsity)":   "SparseGPT 175B (4:8 Sparsity)",
    "SparseGPT (175B, 50% Sparsity)":   "SparseGPT 175B (50% Sparsity)",
}

# ─────────────────────────────────────────────────────────────────────────────
# RENAME-THEN-REMAP: first rename source to target, then remap others into it.
# The RENAME creates the canonical name; subsequent REMAPs collapse siblings.
# Order matters: renames are processed first.
# ─────────────────────────────────────────────────────────────────────────────
RENAME_CREATE = {
    # Claude 3.5 Sonnet: the dated version becomes the canonical; others remap into it.
    "Claude 3.5 Sonnet (20240620)": "Claude 3.5 Sonnet",
}


# ─────────────────────────────────────────────────────────────────────────────
# THINKING-REMAP: old_id → (canonical_id, reasoning_enabled)
# Results get reasoning_enabled=True; model_name updated to canonical.
# ─────────────────────────────────────────────────────────────────────────────
THINKING_REMAP = {
    # (20250514, extended thinking) suffix — map to canonical without date
    "Claude 4 Opus (20250514, extended thinking)":   ("Claude Opus 4",    True),
    "Claude 4 Sonnet (20250514, extended thinking)": ("Claude Sonnet 4",  True),
    # (thinking) suffix — base already exists
    "Claude Haiku 4.5 (thinking)":   ("Claude Haiku 4.5",   True),
    "Claude Opus 4.5 (thinking)":    ("Claude Opus 4.5",    True),
    "Claude Sonnet 4.5 (thinking)":  ("Claude Sonnet 4.5",  True),
}


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

def validate(m_df: pd.DataFrame) -> bool:
    existing = set(m_df["model_id"])
    # Targets that will exist after RENAME-CREATE and RENAME steps
    will_exist = existing | set(RENAME_CREATE.values()) | set(RENAME.values())
    ok = True
    for old, new in REMAP.items():
        if new and new not in will_exist:
            print(f"  ERROR REMAP target missing: {old!r} → {new!r}")
            ok = False
    for old, (new, _) in THINKING_REMAP.items():
        if new not in will_exist:
            print(f"  ERROR THINKING_REMAP target missing: {old!r} → {new!r}")
            ok = False
    if ok:
        print("  All targets verified ✓")
    return ok


def apply(m_df: pd.DataFrame, r_df: pd.DataFrame, write: bool):
    m = m_df.copy()
    r = r_df.copy()

    def existing_ids():
        return set(m["model_id"])

    # ── 1. REMOVE ─────────────────────────────────────────────────────────────
    for mid, reason in REMOVE.items():
        if mid not in existing_ids():
            print(f"  WARN REMOVE not found: {mid!r}")
            continue
        n = (r["model_name"] == mid).sum()
        print(f"  REMOVE  {mid!r}  ({n}r)  — {reason}")
        if write:
            m = m[m["model_id"] != mid]
            r = r[r["model_name"] != mid]

    # ── 2. RENAME-CREATE: rename to create canonical targets ──────────────────
    for old, new in RENAME_CREATE.items():
        if old not in existing_ids():
            print(f"  WARN RENAME-CREATE src not found: {old!r}")
            continue
        if new in existing_ids():
            # target already exists — treat as REMAP
            n = (r["model_name"] == old).sum()
            print(f"  RENAME-CREATE→REMAP  {old!r} -> {new!r}  ({n}r)")
            if write:
                r.loc[r["model_name"] == old, "model_name"] = new
                r.loc[r["model_id"]   == old, "model_id"]   = new
                m = m[m["model_id"] != old]
        else:
            n = (r["model_name"] == old).sum()
            print(f"  RENAME-CREATE  {old!r} -> {new!r}  ({n}r)")
            if write:
                m.loc[m["model_id"] == old,   "model_id"]   = new
                m.loc[m["model_name"] == old,  "model_name"] = new
                r.loc[r["model_name"] == old,  "model_name"] = new
                r.loc[r["model_id"] == old,    "model_id"]   = new

    # ── 3. RENAME ─────────────────────────────────────────────────────────────
    for old, new in RENAME.items():
        if old not in existing_ids():
            print(f"  WARN RENAME src not found: {old!r}")
            continue
        if new in existing_ids():
            n = (r["model_name"] == old).sum()
            print(f"  RENAME→REMAP  {old!r} -> {new!r}  ({n}r)")
            if write:
                r.loc[r["model_name"] == old, "model_name"] = new
                r.loc[r["model_id"]   == old, "model_id"]   = new
                m = m[m["model_id"] != old]
        else:
            n = (r["model_name"] == old).sum()
            print(f"  RENAME  {old!r} -> {new!r}  ({n}r)")
            if write:
                m.loc[m["model_id"] == old,   "model_id"]   = new
                m.loc[m["model_name"] == old,  "model_name"] = new
                r.loc[r["model_name"] == old,  "model_name"] = new
                r.loc[r["model_id"] == old,    "model_id"]   = new

    # ── 4. REMAP ──────────────────────────────────────────────────────────────
    for old, new in REMAP.items():
        if old not in existing_ids():
            print(f"  WARN REMAP src not found: {old!r}")
            continue
        if new not in existing_ids():
            print(f"  ERROR REMAP target still missing: {old!r} → {new!r}")
            continue
        n = (r["model_name"] == old).sum()
        print(f"  REMAP  {old!r} -> {new!r}  ({n}r)")
        if write:
            r.loc[r["model_name"] == old, "model_name"] = new
            r.loc[r["model_id"]   == old, "model_id"]   = new
            m = m[m["model_id"] != old]

    # ── 5. THINKING-REMAP ─────────────────────────────────────────────────────
    for old, (new, reasoning) in THINKING_REMAP.items():
        if old not in existing_ids():
            print(f"  WARN THINKING-REMAP src not found: {old!r}")
            continue
        if new not in existing_ids():
            print(f"  ERROR THINKING-REMAP target missing: {old!r} → {new!r}")
            continue
        n = (r["model_name"] == old).sum()
        print(f"  THINKING-REMAP  {old!r} -> {new!r}  reasoning_enabled=True  ({n}r)")
        if write:
            mask = r["model_name"] == old
            r.loc[mask, "reasoning_enabled"] = "True"
            r.loc[mask, "model_name"] = new
            r.loc[r["model_id"] == old, "model_id"] = new
            m = m[m["model_id"] != old]

    # ── 6. Missed setup extraction ────────────────────────────────────────────
    # Recompute existing ids after all renames/remaps
    # Exclude entries already handled by THINKING_REMAP to avoid double-processing
    cur_ids = existing_ids() if write else set(m_df["model_id"])
    entries = [e for e in detect_missed_setups(cur_ids)
               if e[0] not in THINKING_REMAP]

    from collections import defaultdict
    by_base = defaultdict(list)
    for old, base, setup_val, is_think, is_no_think in entries:
        by_base[base].append((old, setup_val, is_think, is_no_think))

    for base, variants in sorted(by_base.items()):
        base_exists = base in existing_ids()
        for old, setup_val, is_think, is_no_think in variants:
            n = (r["model_name"] == old).sum() if not write else 0
            if not write:
                n = (r_df["model_name"] == old).sum()
                flag = " [reasoning_enabled=True]" if is_think else (
                       " [setup=thinking disabled]" if is_no_think else
                       f" [setup={setup_val!r}]" if setup_val else "")
                op = "REMAP" if base_exists else "RENAME→CREATE"
                print(f"  SETUP-{op}  {old!r}")
                print(f"           -> {base!r}{flag}  ({n}r)")
            if write:
                mask = r["model_name"] == old
                # Set setup or reasoning_enabled
                if is_think:
                    r.loc[mask, "reasoning_enabled"] = "True"
                elif is_no_think:
                    r.loc[mask & (r["setup"] == ""), "setup"] = "thinking disabled"
                elif setup_val:
                    r.loc[mask & (r["setup"] == ""), "setup"] = setup_val
                # Update FK
                r.loc[mask, "model_name"] = base
                r.loc[r["model_id"] == old, "model_id"] = base
                # Handle model entry
                if base in existing_ids():
                    m = m[m["model_id"] != old]
                else:
                    m.loc[m["model_id"] == old,   "model_id"]   = base
                    m.loc[m["model_name"] == old,  "model_name"] = base
                    # refresh
                    cur_ids = set(m["model_id"])

    # ── 7. Post-merge dedup ───────────────────────────────────────────────────
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
        sys.exit(1)

    print(f"\n{'Applying' if args.write else 'Dry run —'} changes:\n")
    m_new, r_new, n_dupes = apply(m_df, r_df, write=args.write)

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
        entries = detect_missed_setups(set(m_df["model_id"]))
        print(f"\nDry run summary:")
        print(f"  REMOVE:         {len(REMOVE)}")
        print(f"  RENAME-CREATE:  {len(RENAME_CREATE)}")
        print(f"  RENAME:         {len(RENAME)}")
        print(f"  REMAP:          {len(REMAP)}")
        print(f"  THINKING-REMAP: {len(THINKING_REMAP)}")
        print(f"  SETUP-extract:  {len(entries)} additional setup-in-name models")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
