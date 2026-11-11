# Metric orientation & scale audit (ICLR 2027 revision)

Response to the review TODO *"check score normalization (lower/higher is better) and scale
standardization"*, raised by all three reviewers: rXDF ("rerun the main analyses after orienting
metrics"), 8Mu7 ("sign-flip issue undercuts the clustering claim"), and XgsZ. Audit date:
2026-11-11, against `data/text_only/` at commit `10f1d3e`.

## TL;DR

| # | Issue | Real? | Affects | Fix (implemented, opt-in) |
|---|---|---|---|---|
| 1 | Lower-is-better metrics never oriented | **Yes.** 8 of 124 columns in S-std, 6–12 per densified matrix | omega_h/omega_t, cosine label cohesion, silhouette, pair separation, viewer map, `default` imputer | `LOWER_IS_BETTER_BENCHMARKS` → negate |
| 2 | Within-column scale corruption (×100 rule) | **Yes.** `the_pile` (in all 6 densified matrices) | Everything touching that column | `SCALE_CORRECTIONS` → ÷100 |
| 3 | Within-column source convention offsets | **Yes.** `bbh`, `math` (same mechanism as the existing `gpqa` fix) | Those two columns | `REVISION_SOURCE_SCALE_CONFLICTS` |
| 4 | Different scales *across* columns (0–1 vs 0–100 vs Elo) | **No.** | Nothing in the EFA | No change. Every imputer z-scores columns or works on correlations |

## 1. How the pipeline treats scale and sign

Mapped by tracing each consumer:

- **Cross-column scale is a non-issue.** `softimpute` uses `biScale(col.center, col.scale)`, `knn`
  standardizes first, and the correlation imputers (`softimpute_corr`, `default`, `zeros`,
  `optspace`, `usvt`, `cvxr`, `ggm`) work on `cor()`. `missForest` trees are scale-invariant, and
  held-out R² is computed in z-space. Pearson correlation and EFA are invariant to any per-column
  affine map, so the 0–100 normalization never mattered for the factor analysis, and Elo, Brier and
  0–1 columns are fine as they are. What does matter is that every row *within* a column uses the
  same scale (§3, §4).
- **Column sign (orientation).** Nothing in `scripts/`, `src/` or `viewer/` reads
  `metric_lower_is_better`.
  - **Sign-equivariant (negating a column only negates its loading row):** all imputers except
    `default`, plus minres+promax EFA and parallel analysis. The factor count and loading magnitudes
    are therefore unchanged by orientation.
  - **Sign-sensitive:**
    - `default` (fill-mean): fills never-co-observed pairs with the *signed* mean off-diagonal r
      (`corr_common.R:189-192`).
    - omega_h / omega_t / omega_group: `psych::omega(..., flip = FALSE)` at `factoring.R:184,200,217,231`.
      Verified in psych's `R/omega.R`: `gsq <- (sum(gload))^2` over `Vt <- sum(m)`, i.e. the
      general-factor share of the variance of the unit-weighted **sum score**. That's the estimand
      rXDF asks about, and it is not average indicator variance. psych defaults to `flip = TRUE`;
      the repo turns that off, so a reversed item's negative g-loading cancels part of the
      numerator. The same goes for `omega_loco.py` delta-omega and its freq-vs-g correlation.
      (psych also accepts `key = ±1` to reflect items by a *given* orientation. That's a useful
      cross-check, but it doesn't fix the cosines or `default`, so the data-level fix is preferred.)
      ECV, the indicator-level measure rXDF requests (Σg² / Σ all squared SL loadings), is computed
      nowhere today and is orientation-invariant.
    - Cosine distance on the signed bifactor rows (`viewer/compute_positions.py:371-375`): a flipped
      column maps d → 2−d against every unflipped column. This feeds `label_cohesion`,
      `label_silhouette`, `pair_separation`, `plot_label_map` and `positions.json`.
  - **Not sensitive:** `top_g*.py` rank by |g|, and `compare_loadings` uses |congruence| per factor.
- `psych::omega(flip = TRUE)` is **not** a valid fix. It reflects items by the sign of their
  g-loading, which is orientation by correlation and builds a positive manifold into the data.

## 2. Orientation: which columns

The row flag `metric_lower_is_better` cannot be used. It is blank on 2,253 of 13,251 rows (every
perplexity benchmark) and wrong on several: ProphetArena Brier, maliciousinstruct/jailbreakbench
ASR and SocialStigmaQA are all flagged `False`, and SpiralBench carries both values. The direction
is therefore curated per benchmark from documented semantics in `config.LOWER_IS_BETTER_BENCHMARKS`
(27 benchmarks). Each entry is tagged `capability` (BPB, perplexity, error, MAE, Brier) or `harm`
(toxicity, LCS-copyright, attack success rate, …). Three more benchmarks are genuinely ambiguous and
listed in `AMBIGUOUS_DIRECTION_BENCHMARKS`: HELM disinformation ×2 (Self-BLEU is a *diversity*
measure) and StrongREJECT.

Columns that survive densification and are reversed:

| Matrix | Reversed columns |
|---|---|
| S std (124) | the_pile, ice, twitter_aae, bold, real_toxicity_prompts, copyright_text (+2 ambiguous disinformation) |
| C std (78) | the_pile, bold, real_toxicity_prompts, copyright_text (+2) |
| R std (298) | S-std set + maliciousinstruct (+2) |
| S agg (293) | R-std set + ProphetArena, sportsmetrics (+2) |
| R agg (310) | S-agg set + pwc_ag_news (+2) |

**Directions checked against source documentation:**

- `cogbench` "Probabilistic Reasoning" (in S-std) is **higher-is-better**, recorded in
  `CHECKED_HIGHER_IS_BETTER`. It is CogBench's *performance* metric. The paper (arXiv 2402.18225)
  App. B.1.4 says: "Performance: Calculated as the posterior accuracy, therefore 1 minus the Bayes
  optimal". It is not a behavioural metric (those are prior/likelihood weighting, which have no
  "better" direction).

**Still unverified** (currently treated as higher-is-better):

- `kaggle_jonlipovetz_chess_suite` (154–1395) and `kaggle_jonlipovetz_game_arena` (2.97–363),
  both in the R matrices.
  - The only record is `docs/CHANGELOG.md` (Kaggle merge): "Chess Suite and Game Arena carry
    Elo-style scores >100". That is the extractor's reading of the leaderboard, not a quote from
    the benchmark's own documentation.
  - The public pages now return 404 when not logged in.

Do not settle these by looking at which models score highest.

**Orientation follows documented semantics, never correlation sign.** `copyright_text` is the
example: lower LCS is better, yet it correlates **+0.21** with the rest of the matrix (stronger
models memorize more). Once oriented it correlates negatively, and that is the result to report.

## 3. Scale bug: `the_pile`

The extraction rule "if 0 ≤ v ≤ 1, ×100" (`docs/extraction_guide.md` §3.1) exempts only the exact
names `BPB`/`Bits/byte`/…, so PwC's `Bits per byte` fractions were stored ×100 (55.7–96.31). The
values > 1 (GPT-2, 1.05–1.23) were left alone, and HELM's `BPB` (0.47–1.61) shares the column.
Verified against the published tables:

- Pile paper: GPT-3 Davinci 0.7177 (stored 71.77), Ada 0.9631 (stored 96.31)
- GLM-130B paper: GLM 0.634 (stored 63.4)

Fix: divide those 22 rows by 100, which keeps all models. `pwc_text8`/`pwc_enwik8` (BPC 98 / 93)
show the same pattern but do not survive densification. They are not corrected, because the
published values were not checked.

Other columns with values ≤ 1 next to values > 1 were checked and are **genuine**:

- BrowseComp: 0.08 = 1/1266 expressed in %
- FrontierMath: 1–2 %
- xnli/xcopa: per-language format failures
- gsm8k/gpqa: zeros

## 4. Source-convention offsets

Every surviving multi-source column was checked for per-source ranges and for shared models. Only
two have a mechanism, the same one as `gpqa`: Open LLM Leaderboard v2 publishes **normalized** BBH
and the **MATH Level-5 subset**.

- `bbh`: v2 median 22.0 vs llm-stats 86.8 (7 rows dropped)
- `math`: v2 median 5.2 vs Vellum/official papers 71–82 (14 rows dropped)

No model appears in both sources, so the offset can't be calibrated. The gsm/gsm8k/math-CoT gaps
(HELM Classic vs Lite, OLL v1 vs PwC) have no mechanism and look like population differences
(2022-era vs newer models), so they are left alone.

## 5. Implementation

`scripts/make_oriented_copy.py` reads `data/text_only/` (left untouched; `make_text_only_copy.py
--check` still passes) and writes a sibling root. It keeps `score_raw` and records each change in
`score_transform`.

```bash
uv run python scripts/make_oriented_copy.py                  # fixes + documented orientation -> data/text_only_oriented
uv run python scripts/make_oriented_copy.py --orient none    # fixes only (isolates the bug-fix effect)
uv run python scripts/make_oriented_copy.py --orient all     # + ambiguous columns (sensitivity)
make preproc DATA_ROOT=data/text_only_oriented DENSITY=0.1   # -> data/text_only_oriented/0.1/
# then the usual docker impute/factor/loco + python post-analyses with
#   DATA_ROOT=data/text_only_oriented RESULTS_ROOT=results/text_only_oriented DENSITY=0.1
# (python scripts take --density 0.1; repeat with DENSITY=0.2 for the paper's second density)
```

The densified shapes are unchanged apart from 1–2 models (S std 669 → 667 × 124).

## 6. Expected effect: preliminary, from observed pairwise correlations

These numbers are **not** the pipeline's loading-space cosines; they are mean within-label
pairwise-complete Pearson r on the densified matrices. The definitive numbers need the R rerun.

| Label (S std) | n | original | fixes only | fixes + orient |
|---|--:|--:|--:|--:|
| task:likelihood_probe | 4 | 0.041 | 0.154 | **0.725** |
| subj:language_modelling | 3 | 0.415 | **0.879** | 0.879 |
| cat:General Knowledge | 21 | 0.376 | 0.360 | **0.555** |
| subj:language_processing | 26 | 0.248 | 0.239 | 0.362 |
| cat:Alignment & Safety | 13 | 0.218 | 0.218 | 0.309 |
| subj:toxicity | 6 | 0.466 | 0.466 | 0.405 |
| *all pairs, mean r* | 124 | 0.390 | 0.389 | 0.401 |

- **Global:** the mean inter-benchmark r barely moves (+0.01), because only ~6 % of columns flip.
  How much omega_h moves is for the rerun to decide.
- **Local:** label cohesion moves a lot for labels that contain reversed members. For capability
  labels the change is consistently upward.
- **Harm labels:** orientation can lower cohesion. In S-aggressive, Alignment & Safety goes
  0.028 → −0.035 and bias_fairness 0.244 → −0.177, because a lower toxicity/bias rate is not a
  capability. Report these with the capability/harm split (`kind` field) rather than hiding them.

## 7. Rerun checklist for the rebuttal

1. Run impute → factor → loco on `text_only_oriented_none` and `text_only_oriented` (all methods;
   `default` must be re-imputed). Optionally add `_all`.
2. Re-run `label_cohesion`, `label_silhouette`, `pair_separation`, `top_g_ci`, `omega_loco`,
   `factor_summary` and `viewer/compute_positions.py`, pointed at each results root.
3. Report the three-way table: omega_h, omega_t, label-cohesion z, and top-g ranking overlap (the
   reviewers' "how much do they change / which conclusions remain stable").
4. **Shortcut, if the original `results/` exist:** for every imputer except `default`, the oriented
   solution equals the original with the reversed rows negated. Negate those rows of the saved
   bifactor loadings and recompute the cosines, then recompute omega on D·R·D. The bug-fix arm
   (`the_pile`, `bbh`, `math`) still needs a real rerun.
5. **Follow-ups:**
   - Correct the_pile in the archive (`data/results.csv`); it's our transcription error, not the
     source's.
   - Change the extraction guide's ×100 rule so the scale is decided per (source, metric) column,
     not per value.
   - Once the revision is accepted, fold these knowledge bases into `make_text_only_copy.py`.
