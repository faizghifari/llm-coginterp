
# Backlog

## 1. Expand dataset: add all pending benchmarks

33 benchmarks are waiting in `notes/pending_benchmarks.md`. For each one: add a row to `benchmarks.csv`, collect and add result rows to `results.csv` with model names refer to `models.csv` if available (if not then add new model row there), then remove it from `pending_benchmarks.md`.

Current pending list by cluster:

**Social behavior / psychology**
- DiploBench

**Games and strategic reasoning**
- Readable Minds, LudoBench, PokerBench, gg-bench

**Humor and creativity**
- CS4, A Confederacy of Models

**Morality, ethics, and values**
- ETHICS, Value Kaleidoscope, Delphi

**Music**
- MSU-Bench, ABC-Eval, WildScore, MASSIVE Music Evaluation, MuchoMusic

**Sports**
- SportQA, SportU, SportsMetrics

**Deception / manipulation**
- Sycophancy Is Not One Thing

**History, culture, and time**
- Counterfactual Reasoning Eligibility

**Cross-cultural**
- Cultural Variations in Moral Judgments

**Specific weird domains**
- EngiBench, KernelBench, BioNovice Lab Performance

**Safety, red-teaming, and robustness**
- MaliciousInstruct, DAN
~~XSTest ✓~~ ~~TrustLLM ✓~~ ~~HarmBench ✓~~ ~~JailbreakBench ✓~~ ~~Do-Not-Answer ✓~~

**Coding and debugging**
- CRITIQUE, DebugBench

After adding: run `python3 verify_data.py` and `python3 export_eee_jsonl.py`.

---

## ~~2. HF Open LLM Leaderboard cleanup~~ ✓ Done

v1 and v2 leaderboard rows are retained as-is. v2 was collected with the "Only Official Providers" filter; v1 did not have an equivalent filter but the model set was reviewed and determined to be acceptable — a mix of official institution models and well-known community/research orgs. No further cleanup needed.

---

## 3. ~~Fill missing quantitative scores~~ ✓ Done

21 of 57 blank-score rows were filled. The remaining 36 are intentionally blank with notes explaining why:
- **moral-machine-llms** (19 rows): MIS scores only in Figure 2a (visual), not extractable as text from the paper.
- **aita-normative** (7 rows): No scalar performance metric exists; the benchmark reports verdict distributions and Krippendorff's alpha, not accuracy/F1.
- **ProphetArena** (3 rows): Scores in Table 6 (appendix); table not rendered in HTML version of paper.
- **moralbench** (2 rows): Model mismatches — paper evaluates Zephyr-7B and Gemma-1.1-7B, not the larger variants in our rows.
- **MenatQA** (2 rows): Model mismatches — paper evaluates Llama-1 variants, rows reference Llama-2. Source URL corrected to `https://arxiv.org/abs/2310.05157`.
- **HistRev/GPT-4o-mini** (1 row): Paper uses GPT-4.1-mini (distinct model); score not attributed.
- **MoralMachine/Meta-Llama-3-70B-Instruct** (1 row): Paper evaluates Llama-3.3-70B-Instruct (different model).
- **ReasonBENCH/Gemini 2.5 Pro** (1 row): Model released after paper publication; not evaluated.
