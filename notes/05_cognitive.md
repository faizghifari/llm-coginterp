# Cognitive Science LLM Benchmarks

Benchmarks designed to evaluate LLM cognitive abilities from a cognitive science perspective — theory of mind, working memory, executive function, metacognition, abstract reasoning, and other capacities studied in cognitive psychology and neuroscience.

---

## ToMBench — Theory of Mind Benchmark (`ToMBench`) — 2024

**Paper:** [ToMBench: Benchmarking Theory of Mind in Large Language Models](https://aclanthology.org/2024.acl-long.847/)  
**Venue:** ACL 2024  
**Languages:** English, Chinese (bilingual)  
**Tasks:** 8 tasks across 31 social cognition abilities  
**Metrics:** Accuracy (MCQ)  

### Description
The most comprehensive ToM benchmark for LLMs, covering 8 tasks and 31 distinct abilities in social cognition. Uses MCQ format to enable automated evaluation. Built from scratch to strictly avoid data leakage from pretraining corpora.

### Key Tasks Covered
- False belief tasks (classic ToM: 1st and 2nd order)
- Faux pas recognition
- Strange stories comprehension
- Hinting task (implicit intent inference)
- Emotion recognition from context
- Social interaction understanding
- Pragmatic inference

### Methodology
Bilingual inventory developed from scratch (not translated from existing English sources). MCQ format avoids generation evaluation variability. 10 popular LLMs evaluated.

### Key Results
| Model | Performance vs. Human |
|---|---|
| GPT-4 | >10 percentage points below human |
| All 10 LLMs evaluated | Below human baseline |

### Cognitive Science Relevance
Theory of Mind is arguably the most theoretically important cognitive capacity for evaluating LLMs from a cognitive science perspective. ToM underlies pragmatic language use, social reasoning, cooperation, and deception detection. The consistent failure of LLMs (even GPT-4) to reach human-level ToM performance suggests fundamental limits in how LLMs represent mental states — they may approximate ToM through surface pattern matching rather than genuine mental state attribution.

Key theoretical connections:
- **Dennett's intentional stance**: Do LLMs adopt genuine intentional stances toward agents?
- **Baron-Cohen's mindblindness theory**: LLM ToM failures parallel autism spectrum profiles
- **Wellman's conceptual change theory**: LLMs may lack the conceptual change mechanisms that enable children to develop robust ToM

### Limitations / Caveats
- MCQ format may allow shortcuts that inflate ToM scores (elimination of wrong options without true mental state attribution).
- LLM performance on ToM tasks degrades sharply under minor perturbations (adding irrelevant context, observer knowledge inconsistencies) — suggesting brittle pattern matching rather than robust ToM.
- Bilingual (EN/ZH only) — cultural variation in ToM tasks not covered.

---

## Triangulating LLM Progress through Benchmarks, Games & Cognitive Tests (`TriLLM-Cog`) — 2025

**Paper:** [Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests](https://arxiv.org/abs/2502.14359)  
**Venue:** EMNLP 2025 Findings  
**Languages:** English  
**Methods:** MMLU, BBH (benchmarks) + Signalling Games, Taboo (interactive games) + Working Memory, ToM (cognitive tests)  

### Description
Compares three evaluation paradigms — static benchmarks, interactive games, and cognitive tests — to determine which best captures different aspects of LLM intelligence. Key finding: interactive games better discriminate model quality for social/executive function tasks.

### Methodology
Multi-method evaluation across 3 paradigms. Factor analysis to identify which evaluation types capture which cognitive dimensions.

### Key Results
- Interactive games are **superior to static benchmarks** for discriminating model quality
- Causal and logical reasoning correlate across all evaluation types
- **Executive functions** (planning, inhibition) and **social/emotional skills** (pragmatics, ToM) show stronger associations with games than static benchmarks
- Cognitive tests (working memory, ToM) measure dimensions not captured by MMLU/BBH

### Cognitive Science Relevance
Methodologically fundamental for the entire field: this paper demonstrates that **static benchmarks systematically under-measure** the most cognitively interesting capabilities (executive function, social cognition). This has direct implications for using LLMs as cognitive science subjects — standard benchmark scores are insufficient proxies for human-analogous cognitive capacities. The triangulation approach mirrors how cognitive science itself uses multiple converging methods (behavioral, neuroimaging, computational) to study cognition.

### Limitations / Caveats
- Interactive game evaluation is harder to standardize across models.
- Factor structure of LLM cognitive abilities may differ from the factor structures of human cognition.

---

## Neuropsychologically Grounded Evaluation of LLM Cognitive Abilities (`NeuroCognition`) — 2026

**Paper:** [A Neuropsychologically Grounded Evaluation of LLM Cognitive Abilities](https://arxiv.org/abs/2603.02540)  
**Year:** 2026  
**Models Evaluated:** 156 models (factor analysis)  
**Methods:** Raven's Progressive Matrices; Spatial Working Memory; Wisconsin Card Sorting Test  

### Description
Develops NeuroCognition — a benchmark based on three adapted clinical neuropsychological assessments — to evaluate foundational cognitive abilities in LLMs beyond task completion.

### Tests Used
1. **Raven's Progressive Matrices (RPM)**: Measures abstract relational reasoning via matrix completion patterns. Classic measure of fluid intelligence (Gf) in human neuropsychology.
2. **Spatial Working Memory (SWM)**: Evaluates capacity to maintain and systematically search spatial information. Tests spatial WM and strategic planning.
3. **Wisconsin Card Sorting Test (WCST)**: Assesses cognitive flexibility (set-shifting) — ability to switch sorting rules in response to feedback. Sensitive to prefrontal cortex function in humans.

### Key Findings
- LLMs show a **unified general capability factor** across standard benchmarks (similar to human g-factor)
- Models **fail on simple human tasks** that should be easy given overall capability
- Stronger performance on text-based tasks vs. image-based versions of same assessments
- Performance declines with increased task complexity
- Simple human-like strategies yield partial gains; complex reasoning doesn't universally help
- NeuroCognition positively correlates with standard benchmarks while measuring **distinct cognitive abilities**

### Cognitive Science Relevance
This benchmark is directly motivated by cognitive science — using clinical neuropsychological instruments adapted for LLMs. Key theoretical contributions:

- **g-factor in LLMs**: Factor analysis across 156 models revealing whether LLMs have a general intelligence factor analogous to Spearman's g (or separate primary mental abilities as in Thurstone's model)
- **Fluid vs. crystallized intelligence**: RPM tests Gf (abstract reasoning); standard benchmarks primarily test Gc (crystallized knowledge) — dissociation between the two in LLMs reveals important limits
- **Cognitive flexibility**: WCST performance reveals whether LLMs can truly shift mental sets or are locked into trained patterns
- **Working memory capacity**: SWM tests whether LLMs have functional WM or merely simulate WM-like behavior through attention mechanisms

The finding that LLMs fail on "simple human tasks" despite high benchmark scores directly mirrors neuropsychological double dissociations — patients who can perform complex tasks but fail simpler ones due to specific cognitive deficits.

### Limitations / Caveats
- Clinical tests adapted for LLMs may not measure exactly the same constructs as in human neuropsychology.
- 156 models in factor analysis provides broad coverage but results depend on model selection.
- Text-based versions of spatial tasks may measure verbal-spatial translation ability rather than spatial cognition per se.

---

## Multilingual ToM Benchmarks — XToM and Multi-ToM

### XToM
Tests theory of mind across multiple languages in parallel, culturally-nuanced settings. Reveals stable **fact retrieval** but substantial **variability in ToM reasoning** across languages — suggesting that ToM task performance is language-mediated.

### Multi-ToM  
Multilingual parallel ToM benchmark. Together with XToM, demonstrates that cross-lingual ToM performance gaps exist beyond what data scarcity alone can explain — cultural pragmatic conventions affect ToM task interpretation.

### Cognitive Science Relevance
Cross-lingual ToM benchmarks are critical for distinguishing between:
1. Language-independent mental state attribution (would transfer uniformly across languages)
2. Language-mediated social reasoning (would show cross-lingual variation)

Current evidence favors interpretation 2: ToM performance varies across languages even when controlling for data availability — supporting linguistic relativity effects in social cognition.

---

## Cross-Cutting Findings: LLM Cognition

### 1. The g-Factor Problem
Standard benchmark scores cluster into a single general capability factor (g-like) in LLMs (NeuroCognition, 156-model analysis). This suggests that current evaluation metrics are largely measuring one general capability rather than differentiating cognitive profiles — a major limitation for cognitively-motivated LLM analysis.

### 2. Static Benchmarks Miss Key Capacities
The Triangulating study shows that executive functions and social skills are better captured by interactive evaluation than static benchmarks. Standard benchmarks like MMLU and BBH primarily measure declarative knowledge and basic reasoning — not the dynamic, socially-situated cognitive processes most interesting from a cognitive science perspective.

### 3. Brittle Cognitive Capacities
- ToM abilities degrade sharply under minor perturbations (ToMBench + literature review)
- CoT reasoning on BBH collapses without explicit scaffolding
- Safety refusal (SORRY-Bench) breaks down under linguistic mutations

These brittleness patterns suggest LLMs implement **surface-level approximations** of cognitive processes rather than robust underlying representations — analogous to "narrow" vs. "broad" cognitive competencies.

### 4. Inverse Scaling in Truthfulness
TruthfulQA's inverse scaling finding — larger models are less truthful — suggests that scale improves mimicry of human text patterns (including misconceptions) rather than improving epistemic accuracy. This mirrors cognitive science findings about how cultural transmission amplifies misinformation.

### 5. ToM Persistently Below Human
Across all ToM benchmarks (ToMBench, XToM, Multi-ToM), no current LLM reaches human-level performance. This persistent gap is theoretically significant: if ToM is fundamental to human social cognition and language use, LLMs' ToM deficit may explain many of their pragmatic language failures (indirect speech, sarcasm, implicature, humor).

---

## CogBench: A Large Language Model Walks into a Psychology Lab (`CogBench`) — 2024

**Paper:** [CogBench: a large language model walks into a psychology lab](https://arxiv.org/abs/2402.18225)  
**Venue:** ICML 2024  
**GitHub:** https://github.com/juliancodaforno/CogBench  
**Authors:** Julian Coda-Forno, Marcel Binz, Jane X. Wang, Eric Schulz  
**Languages:** English  
**Models Evaluated:** 35 LLMs (proprietary + open-source)  
**Metrics:** 10 human-normalized behavioral metrics (0 = random agent, 1 = average human)  

### Description
The most comprehensive cognitive psychology battery applied to LLMs to date. Adapts 7 canonical experiments from cognitive psychology and measures 10 behavioral metrics across 35 LLMs. Uses statistical multilevel regression to identify which model properties (size, RLHF, open-source status, code fine-tuning) predict human-like cognitive behavior.

### Seven Experiments & Ten Behavioral Metrics

| Experiment | Source | Metric(s) | What it measures |
|---|---|---|---|
| **Probabilistic Reasoning** | Dasgupta et al., 2020 | Prior weighting, Likelihood weighting | Belief updating from evidence |
| **Horizon Task** | Wilson et al., 2014 | Directed exploration, Random exploration | Explore/exploit tradeoff in multi-armed bandit |
| **Restless Bandit** | Ershadmanesh et al., 2023 | Meta-cognition | Confidence calibration in non-stationary rewards |
| **Instrumental Learning** | Lefebvre et al., 2017 | Learning rate, Optimism bias | RL from positive/negative feedback |
| **Two-Step Task** | Daw et al., 2011 | Model-basedness | Model-based vs. model-free reinforcement learning |
| **Temporal Discounting** | Ruggeri et al., 2022 | Temporal preference | Delay discounting (impulsivity vs. patience) |
| **BART (Balloon Risk Task)** | Lejuez et al., 2002 | Risk-taking | Risk tolerance under uncertainty |

### Models Evaluated
**Proprietary:** GPT-4, text-davinci-003, text-davinci-002, Claude-1, Claude-2, PaLM-2 (text-bison)  
**Open-source:** Falcon-40B, MPT-30B, LLaMA-2 (7B/13B/70B + chat), Vicuna (7B/13B), LongAlpaca, CodeLlama  

### Key Results

| Finding | Details |
|---|---|
| GPT-4 & Claude-1 | Reach human-level on 5 of 6 performance metrics |
| All 35 models | Show strong **optimism bias** in instrumental learning |
| All models | **Overweight priors** in probabilistic reasoning (system neglect) |
| Most models | **Lack meta-cognition** (exceptions: GPT-4, Claude-1) |
| GPT-4 | **Exceeds human** on Two-Step Task (model-based reasoning) |
| All models (Horizon) | Achieve super-human performance through exploitation only — lack directed exploration |
| BART (Risk) | Models cluster at extremes: never or always risk-taking; no nuanced risk sensitivity |
| text-bison | Only model NOT outperforming humans on the horizon task |

### Regression Analysis Results (Multi-level)

| Predictor | Effect on | β | p |
|---|---|---|---|
| Model size | Performance | +0.277 | <0.001 |
| Model size | Model-basedness | +0.481 | <0.001 |
| RLHF | Meta-cognition | +0.461 | <0.001 |
| RLHF | Human-likeness (L2) | −11.7% distance | — |
| Open-source | Risk-taking | −0.612 | <0.001 |
| Code fine-tuning | Performance/behavior | ~0 (n.s.) | n.s. |

### Prompt Engineering Effects
- Chain-of-Thought (CoT): +9% probabilistic reasoning, **+64.6% model-basedness**
- Take-a-Step-Back (SB): +3.1% probabilistic reasoning, **+118.6% model-basedness**

### Cognitive Science Relevance
CogBench is theoretically grounded in computational cognitive neuroscience — each task taps a distinct cognitive system:

- **Probabilistic Reasoning**: Tests Bayesian belief updating — whether LLMs implement ideal observer models or show human-like biases (base rate neglect, overconfidence)
- **Horizon Task**: Tests the **explore/exploit dilemma** — a fundamental problem in learning theory. LLMs' lack of directed exploration suggests they optimize for immediate expected value rather than information gain
- **Restless Bandit + Meta-cognition**: Tests whether LLMs have second-order representations of their own uncertainty — a prerequisite for genuine metacognition
- **Instrumental Learning + Optimism Bias**: The universal optimism bias (models learn faster from positive feedback) mirrors a well-documented human cognitive bias with roots in dopaminergic learning systems
- **Two-Step Task**: The gold standard for distinguishing **model-based** (goal-directed, flexible) vs. **model-free** (habitual, inflexible) reinforcement learning in humans. GPT-4 exceeding human model-basedness is a striking finding
- **Temporal Discounting**: Tests impulsivity and future-orientation — core dimensions of human decision-making with well-established neuroscientific correlates
- **BART**: Tests **risk sensitivity calibration** — LLMs' extreme (never/always) patterns suggest no sensitivity to outcome variance, unlike human risk preferences

The RLHF finding is particularly important: RLHF training substantially increases human-likeness across behavioral metrics — suggesting that human feedback alignment influences cognitive behavioral patterns beyond just output quality.

### Limitations / Caveats
- Temperature = 0 (deterministic) — human participants show stochastic behavior; this limits behavioral comparability
- Reduced trial counts due to context length constraints (e.g., 4 blocks vs. 20 human blocks in restless bandit)
- External validity of these laboratory tasks for real-world LLM behavior is unestablished
- Contamination possible (though procedural generation minimizes this)
- Most models are pre-2024; newer models (GPT-4o, Claude 3.5, etc.) not evaluated

---

## Using Cognitive Psychology to Understand GPT-3 (`CogPsych-GPT3`) — 2022/2023

**Paper:** [Using cognitive psychology to understand GPT-3](https://arxiv.org/abs/2206.14576)  
**Venue:** PNAS 2023  
**Authors:** Marcel Binz, Eric Schulz  
**Languages:** English  
**Task:** Battery of canonical cognitive psychology experiments  

### Description
Seminal paper applying cognitive psychology experimental paradigms to an LLM (GPT-3). Directly inspired CogBench. Tests GPT-3 on decision-making, information search, deliberation, causal reasoning, multi-armed bandit, model-based RL, and directed exploration.

### Key Results
| Task | GPT-3 Performance |
|---|---|
| Vignette-based decision making | Matches or exceeds humans |
| Multi-armed bandit | Outperforms humans |
| Model-based RL signatures | Present |
| Directed exploration | Absent (fails) |
| Causal reasoning | Fails ("miserably") |
| Robustness to perturbations | Poor (brittle) |

### Cognitive Science Relevance
First systematic application of cognitive psychology methods to LLMs. Established the framework that CogBench later operationalized at scale. Key finding: GPT-3 has a distinctive **cognitive profile** — not uniformly human-like or unlike, but selectively matching humans on some capacities while failing on others. This profile approach is more theoretically rich than single benchmark scores.

---

## Human-Like Reasoning Biases in LLMs (`ReasoningBiases`) — 2023

**Paper:** [Human-Like Intuitive Behavior and Reasoning Biases Emerged in Language Models -- and Disappeared in GPT-4](https://arxiv.org/abs/2306.07622)  
**Venue:** Nature Computational Science 2023  
**Authors:** Hagendorff, Fabi, Kosinski  
**Tasks:** Cognitive Reflection Test (CRT), semantic illusions  
**Models:** GPT-3, ChatGPT, GPT-4  

### Description
Tests whether LLMs exhibit human-like System 1 (intuitive, error-prone) vs. System 2 (deliberate, accurate) reasoning. Finds that GPT-3 makes the same intuitive errors as humans, while ChatGPT and GPT-4 are "hyperrational."

### Key Results
- **GPT-3**: Shows human-like System 1 biases — makes intuitive errors on CRT and semantic illusions
- **ChatGPT & GPT-4**: Avoid intuitive errors even without chain-of-thought; behave "hyperrationally"
- RLHF + scale appear to eliminate System 1 patterns

### Cognitive Science Relevance
Directly tests **Dual Process Theory** (Kahneman's System 1/2) in LLMs. The finding that RLHF-trained models become hyperrational (no System 1 errors) is ambiguous: it could indicate genuine System 2 dominance, or that RLHF training surfaces pattern matching that avoids known bias traps without genuine deliberative reasoning. This connects to debates about whether LLMs can implement anything analogous to deliberate reasoning.

*Note: Paper was revised; some overlap with arXiv:2212.05206.*

---

## Large Language Models Fail on Trivial ToM Alterations (`ToM-Brittleness`) — 2023

**Paper:** [Large Language Models Fail on Trivial Alterations to Theory-of-Mind Tasks](https://arxiv.org/abs/2302.08399)  
**Venue:** arXiv 2023  
**Author:** Tomer Ullman (Harvard)  
**Tasks:** Modified false-belief tasks (Sally-Anne variants with surface perturbations)  
**Models:** GPT-4 and others  

### Description
Directly challenges claims of LLM Theory of Mind success (Kosinski 2023). Shows that small alterations to standard false-belief tasks — while preserving the underlying ToM reasoning requirement — cause LLM performance to collapse to near-chance, suggesting surface pattern matching rather than genuine ToM.

### Key Results
LLMs fail on "trivially altered" ToM tasks. Performance is brittle: small changes to wording, location, or agent names that don't affect the correct answer cause failure.

### Cognitive Science Relevance
This is one of the most theoretically important LLM cognitive papers. It demonstrates that **apparent ToM competence is an artifact of training data overlap** — LLMs have memorized patterns associated with false-belief tasks rather than learning underlying mental state attribution mechanisms. Crucial for the research project's cognitive analysis: LLM "cognitive" performance must be evaluated for robustness, not just average accuracy.

Key connection to cognitive science: Competence in cognitive tasks should generalize to novel variations that preserve the relevant cognitive demand — this is the standard in developmental ToM research (Wimmer & Perner, 1983 paradigm). LLMs fail this test of genuine competence.

---

## Playing Repeated Games with LLMs (`RepeatedGames`) — 2023

**Paper:** [Playing repeated games with Large Language Models](https://arxiv.org/abs/2305.16867)  
**Venue:** Nature Human Behaviour 2025  
**Authors:** Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, Eric Schulz  
**Tasks:** Repeated 2×2 games: Prisoner's Dilemma, Battle of the Sexes, Stag Hunt  
**Models:** GPT-4 (and others vs. each other and human players)  

### Description
Uses behavioral game theory to evaluate LLM social cognition. Models play finitely repeated 2×2 games against each other and against human players.

### Key Results

| Game | GPT-4 Behavior |
|---|---|
| Prisoner's Dilemma | High cooperation initially; **unforgiving defector** after one betrayal |
| Battle of the Sexes | **Suboptimal** — fails coordination despite information about opponent |
| Stag Hunt | Moderate performance |
| vs. Humans (with SCoT) | Improved cooperation and coordination |

Social chain-of-thought (SCoT) prompting significantly improves cooperation and coordination, especially with human players.

### Cognitive Science Relevance
Tests **strategic social cognition** — whether LLMs can reason about other agents' minds and strategies in a dynamic social setting. Key findings:

- **Self-interested games** (Prisoner's Dilemma): LLMs do well — consistent with their strong individual optimization
- **Coordination games** (Battle of the Sexes): LLMs fail — require modeling the other agent's preferences, not just one's own payoffs. This is a joint-attention/ToM demand
- The "unforgiving defector" pattern (cooperate → defect permanently after betrayal) echoes a simplistic Tit-for-Tat-like strategy — effective in evolutionary terms but suboptimal in finite games where defection should be anticipated near the end

Connection to CogBench's social cognition findings and ToMBench's ToM deficits: LLMs struggle when tasks require genuine modeling of other agents' mental states (preferences, beliefs, intentions) rather than individual optimization.

---

## Updated Cross-Cutting Findings (after CogBench additions)

### 6. Distinctive Cognitive Profiles
CogBench + Binz & Schulz together establish that LLMs have **uneven cognitive profiles**: strong at model-based reasoning and vignette tasks, weak at directed exploration, causal reasoning, meta-cognition, and risk calibration. These profiles differ from simple "more capable = more human-like" — each cognitive dimension has its own developmental trajectory.

### 7. RLHF as Cognitive Alignment
CogBench's finding that RLHF decreases L2 distance to human behavior by 11.7% suggests that **human feedback alignment shapes cognitive behavioral patterns**, not just output preferences. This has implications for cognitive science: RLHF may be inadvertently creating models whose decision-making better mimics human cognitive biases (optimism, risk aversion) along with human preferences.

### 8. Optimism Bias is Universal
Every single model in CogBench shows optimism bias (learning faster from positive feedback than negative). This is one of the most robust LLM cognitive findings — it may reflect an asymmetry in how positive vs. negative examples are weighted in pretraining objectives (e.g., next-token prediction on successful completions).

### 9. ToM: Pattern Matching vs. Genuine Competence
Ullman (2023) + ToMBench + XToM convergently show that LLM ToM is brittle and non-generalizing. CogBench adds that even where LLMs show social reasoning (Two-Step Task model-basedness), this is driven by scale and RLHF rather than architectural ToM mechanisms. The Repeated Games study confirms social coordination failures.


---

## Multimodal Temporal-Intent Reasoning Addendum (2026-04)

Several newly-added multimodal benchmarks are highly relevant to cognitive analysis and should be treated as complementary probes:

- TemporalBench: temporal event ordering and fine-grained temporal discrimination.
- NExT-QA: causal-temporal explanation in dynamic scenes.
- IntentQA: latent intention inference in videos.
- R-AVST / V-STaR / MLVU / LongShOT: harder spatio-temporal, cross-modal memory, and long-horizon planning conditions.

These benchmarks are now included in the dataset and primarily documented in `01_multilingual.md` as multimodal evaluation additions.

