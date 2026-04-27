# Alignment and Safety

This document categorizes benchmarks related to alignment and safety.

## Alignment/Safety Benchmarks

### AgentIF: Agentic Instruction Following (`agentif`) [Link](https://arxiv.org/abs/2505.16944)
**Subcategory:** instruction-following

No description available.

### AIR-Bench (`air_bench`) [Link](https://arxiv.org/abs/2402.07729)
**Subcategory:** image-reasoning

No description available.

### AlpacaEval (`alpacaeval`) [Link](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)
**Subcategory:** instruction-following

No description available.

### Arena-Hard-Auto (`arena_hard_auto`) [Link](https://huggingface.co/datasets/lmsys/arena-hard-auto)
**Subcategory:** conversational

No description available.

### BFCL: Berkeley Function Calling Leaderboard (`bfcl`) [Link](https://huggingface.co/datasets/berkeley-call-leaderboard/BFCL)
**Subcategory:** tool-use

No description available.

### Chatbot Arena (`chatbot_arena`) [Link](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
**Subcategory:** conversational

No description available.

### DarkBench (`DarkBench`)
**Subcategory:** dark-content

No description available.

### DeceptionBench (`DeceptionBench`) [Link](https://arxiv.org/abs/2510.15501)
**Subcategory:** deception

No description available.

### DialogBench (`dialogbench`) [Link](https://arxiv.org/abs/2311.01677)
**Subcategory:** conversational

No description available.

### Do-Not-Answer (`do_not_answer`) [Link](https://arxiv.org/abs/2308.13387)
**Subcategory:** safety-evaluation

Dataset of 939 prompts that responsible LLMs should not follow. Evaluates safeguard mechanisms using a 5-level action taxonomy. LLaMA-2-7b-chat safest (0.32% harmful); ChatGLM2 least safe (9.05%).

### EIFBench (`eifbench`) [Link](https://huggingface.co/datasets/evaleval/EIFBench)
**Subcategory:** instruction-following

No description available.

### FEVER (`fever`) [Link](https://huggingface.co/datasets/fever/fever)
**Subcategory:** fact-verification

No description available.

### FollowBench (`followbench`) [Link](https://arxiv.org/abs/2310.20410)
**Subcategory:** instruction-following

No description available.

### GaslightingBench (`GaslightingBench`) [Link](https://arxiv.org/abs/2501.19017)
**Subcategory:** deception

No description available.

### HarmBench (`harmbench`) [Link](https://arxiv.org/abs/2402.04249)
**Subcategory:** red-teaming

Standardized framework for automated red teaming. 510 harmful behaviors across 7 semantic + 4 functional categories. Uses fine-tuned Llama 2 13B classifier (93.19% accuracy). GCG ASR baseline: Zephyr-7B 31.8%.

### Hagendorff Biases (2023) (`hagendorff_biases_2023`) [Link](https://github.com/langchain-ai/langchain/issues/35438)
**Subcategory:** bias-detection

No description available.

### HELM (`helm`) [Link](https://github.com/stanford-crfm/helm/issues/1723)
**Subcategory:** multi-task

No description available.

### IFEval (`ifeval`) [Link](https://huggingface.co/datasets/google/IFEval)
**Subcategory:** instruction-following

No description available.

### JailbreakBench (`jailbreakbench`) [Link](https://arxiv.org/abs/2404.01318)
**Subcategory:** jailbreak-robustness

Open robustness benchmark for jailbreaking LLMs. JBB-Behaviors: 100 harmful + 100 benign across 10 categories. Uses Llama-3-70B as judge (90.7% human agreement). RS attack ASR: Llama-2 90%, GPT-4 78%.

### LawBench (`lawbench`) [Link](https://huggingface.co/datasets/CSRC-CASIA/LawBench)
**Subcategory:** legal-reasoning

No description available.

### LegalBench (`legalbench`) [Link](https://huggingface.co/datasets/nguha/legalbench)
**Subcategory:** legal-reasoning

No description available.

### LegalEval-Q (`LegalEval-Q`) [Link](https://arxiv.org/abs/2505.24826)
**Subcategory:** legal-reasoning

No description available.

### MCEval (`mceval`) [Link](https://arxiv.org/html/2406.07436v1)
**Subcategory:** evaluation

No description available.

### SocialStigmaQA (`SocialStigmaQA`)
**Subcategory:** bias-detection

No description available.

### Sorry-Bench (`sorry_bench`) [Link](https://github.com/SORRY-Bench/sorry-bench)
**Subcategory:** safety-refusal

No description available.

### StrongREJECT (`StrongREJECT`) [Link](https://arxiv.org/abs/2402.10260)
**Subcategory:** safety-refusal

No description available.

### Swiss-LegalBench (`swiss_legal_bench`) [Link](https://github.com/HazyResearch/legalbench/)
**Subcategory:** legal-reasoning

No description available.

### TrustLLM (`trustllm`) [Link](https://arxiv.org/abs/2401.05561)
**Subcategory:** trustworthiness

Comprehensive framework for 8 trustworthiness dimensions. 30+ datasets, 16 models. Llama2-7b 57% benign prompt refusal (over-alignment). GPT-4 80.5% OOD refusal-to-answer rate.

### TruthfulQA (`truthfulqa`) [Link](https://huggingface.co/datasets/domenicrosati/TruthfulQA)
**Subcategory:** truthfulness

No description available.

### Vectara Hallucination Leaderboard (`vectara`) [Link](https://github.com/vectara/hallucination-leaderboard)
**Subcategory:** hallucination

No description available.

### WildBench (`wildbench`) [Link](https://huggingface.co/datasets/allenai/WildBench)
**Subcategory:** instruction-following

No description available.

### XSTest (`xstest`) [Link](https://arxiv.org/abs/2308.01263)
**Subcategory:** safety-refusal

Diagnostic test suite for exaggerated safety and false refusals. 250 safe prompts across 10 categories. Llama2.0 38% refusal on safe prompts; GPT-4 6.4%; Mistral-7B-Instruct 0.8%.

### MaliciousInstruct (`maliciousinstruct`) [Link](https://arxiv.org/abs/2310.06987)
**Subcategory:** malicious-prompt-resistance

Safety/red-teaming benchmark from "Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation" (Huang et al., ICLR 2024). 100 prompts across 10 malicious categories. ASR measured under 6 decoding conditions. Default evaluation shows <5% ASR on aligned models; simple decoding manipulation pushes ASR to >95%. Batch 9 addition — 11 models, 66 result rows.
