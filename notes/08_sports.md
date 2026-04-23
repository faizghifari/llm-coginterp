# Sports

This document categorizes benchmarks related to sports knowledge, understanding, and reasoning.

## Sports Benchmarks

### SportQA (`sportqa`) [Link](https://arxiv.org/abs/2402.15862)
**Subcategory:** sports-knowledge
**Level:** NAACL 2024

Comprehensive sports understanding benchmark with 70,592 MCQs across 35 sports and 3 difficulty levels (foundational, rules/tactics, scenario-based analysis). Level-3 features multi-hop and single-hop questions created manually by intercollegiate athletes and coaches. Accuracy: GPT-4 23.01% (L3), GPT-3.5-Turbo 19.24%, PaLM-2-Bison 16.74%, Llama-2-13B-Chat 8.79%. Human experts score 91.84% on Level-3. GPT-4 remains ~68% less accurate than human experts on hardest tasks.

### SPORTU (`sportu`) [Link](https://arxiv.org/abs/2410.08474)
**Subcategory:** sports-understanding
**Level:** ICLR 2025

Multimodal benchmark evaluating MLLMs on sports understanding across text (900 MCQs with human-annotated explanations) and video (1,701 slow-motion clips across 7 sports: Soccer, Basketball, Volleyball, Ice Hockey, Tennis, Baseball, Badminton). 12,048 QA pairs (10,973 MCQ; 1,075 open-ended) across 3 difficulty levels. Accuracy: Qwen2-VL-72B 70.94% (hard: 44.12%), Claude-3.5-Sonnet 70.18% (hard: 53.06%), GPT-4o 68.79% (hard: 56.20%), LLaVA-NeXT-72B 63.72% (hard: 30.78%). Reasoning before answer hurts accuracy; direct answering outperforms CoT.

### SportsMetrics (`sportsmetrics`) [Link](https://arxiv.org/abs/2402.10979)
**Subcategory:** numerical-reasoning
**Level:** ACL 2024

Evaluates LLM numerical reasoning and information fusion using detailed ESPN play-by-play data (28,492 NBA games; 5,867 NFL games). Four capabilities tested: long-form reasoning, adapting to new rules, robustness against noise, and planning with working memory. MAE on NBA points: GPT-3.5-Turbo-1106 9.45 (best), Gemini Pro 17.62, claude-2.1 21.73, Llama-2-13B-Chat 70.77. Models with context windows >16k significantly outperform standard models. Knowledge conflict when player names are swapped is a key failure mode.
