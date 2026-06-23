#!/usr/bin/env python3
"""
Standardisation Pass 6 — Pre-LLM/Compound Model Sweep
=======================================================
Scope rule: keep general-purpose generative models (LLMs, fine-tuned LLMs,
general language models, multimodal generative models, general seq2seq models).
Remove task-specific discriminative models, classical ML, contrastive/embedding
models, MT-specific systems, KD-training systems, compound pipelines with
separate inference-time models, and pure method/prompting labels.

Operations:
  REMOVE     — cascade delete model + all results rows
  RENAME     — rename model_id in-place (no existing canonical target)
  RENAME_SETUP — set results.setup after RENAME
  REMAP      — merge results into existing canonical, delete old entry
  SETUP_REMAP — extract setup qualifier into results.setup, merge into base

Usage:
  python3 scripts/standardise_pass6.py           # dry run
  python3 scripts/standardise_pass6.py --write   # apply

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
    # ── Discriminative extractive QA models ──────────────────────────────────
    "300D LSTMN with deep attention fusion": "discriminative span reader",
    "300D MMA-NSE encoders with attention": "discriminative encoder",
    "450D LSTMN with deep attention fusion": "discriminative span reader",
    "600D Hierarchical BiLSTM with Max Pooling (HBMP, code)": "discriminative encoder",
    "AttentionReader+ (ensemble)": "discriminative reader",
    "AttentionReader+ (single)": "discriminative reader",
    "AttentiveReader + bilinear attention": "discriminative reader",
    "BiDAF + Self Attention (single model)": "discriminative span reader",
    "BiDAF + Self Attention + ELMo": "discriminative span reader",
    "BiDAF + Self Attention + ELMo (ensemble)": "discriminative span reader",
    "BiDAF + Self Attention + ELMo (single model)": "discriminative span reader",
    "BiDAF + Self Attention + ELMo + A2D (single model)": "discriminative span reader",
    "BiDAF++ (single model)": "discriminative span reader",
    "BiDAF++ with pair2vec (single model)": "discriminative span reader",
    "DCN (Char + CoVe)": "discriminative span reader",
    "DCN+ (ensemble)": "discriminative span reader",
    "DCN+ (single model)": "discriminative span reader",
    "DCN+ (single)": "discriminative span reader",
    "DMN+": "dynamic memory network, discriminative",
    "DMN+ [xiong2016dynamic]": "dynamic memory network, discriminative",
    "DocQA + ELMo": "discriminative reader",
    "DocQA + NeurQuRI (single model)": "discriminative reader",
    "FusionNet++ (ensemble)": "discriminative reader",
    "HBMP + word2vec": "discriminative encoder",
    "HGN + SemanticRetrievalMRS IR": "discriminative reader + retrieval compound",
    "HGN-large + SemanticRetrievalMRS IR": "discriminative reader + retrieval compound",
    "HopRetriever + Sp-search": "discriminative reader + retrieval compound",
    "Interactive AoA Reader+ (ensemble)": "discriminative reader",
    "Interactive AoA Reader+ (single model)": "discriminative reader",
    "KACTEIL-MRC (GF-Net+)": "discriminative reader",
    "KACTEIL-MRC (GF-Net+Distillation)": "discriminative reader",
    "KACTEIL-MRC(GF-Net+) (ensemble)": "discriminative reader",
    "KACTEIL-MRC(GF-Net+) (single model)": "discriminative reader",
    "KACTEIL-MRC(GF-Net+Distillation) (single model)": "discriminative reader",
    "MAMCN+ (single model)": "discriminative reader",
    "Masque (NarrativeQA + MS MARCO)": "discriminative reader",
    "Match-LSTM with Ans-Ptr (Boundary)": "discriminative span reader",
    "Match-LSTM with Ans-Ptr (Boundary) (ensemble)": "discriminative span reader",
    "Match-LSTM with Ans-Ptr (Sentence)": "discriminative span reader",
    "Match-LSTM with Bi-Ans-Ptr (Boundary)": "discriminative span reader",
    "Match-LSTM with Bi-Ans-Ptr (Boundary+Search+b)": "discriminative span reader",
    "QGHC+Att+Concat": "discriminative reader",
    "RaSoR + TR (single model)": "discriminative span reader",
    "RaSoR + TR + LM (single model)": "discriminative span reader",
    "Reinforced Mnemonic Reader + A2D (ensemble model)": "discriminative reader",
    "Reinforced Mnemonic Reader + A2D (single model)": "discriminative reader",
    "Reinforced Mnemonic Reader + A2D + DA (single model)": "discriminative reader",
    "Reinforced Mnemonic Reader + Answer Verifier (single model)": "discriminative reader",
    "SEDT+BiDAF (ensemble)": "discriminative reader",
    "SEDT+BiDAF (single model)": "discriminative reader",
    "SLQA+": "discriminative reader",
    "SLQA+ (ensemble)": "discriminative reader",
    "SLQA+ (single model)": "discriminative reader",
    "Span Extract + Classify (single model)": "discriminative span extractor",
    "Tree-LSTM + BiDAF + ELMo (single model)": "discriminative reader",

    # ── NLI / textual-entailment discriminative models ───────────────────────
    "200D decomposable attention feed-forward model with intra-sentence attention": "discriminative NLI",
    "200D decomposable attention model with intra-sentence attention": "discriminative NLI",
    "600D ESIM + 300D Syntactic TreeLSTM": "discriminative NLI",
    "Bi-Attention + DCU-LSTM": "discriminative NLI",
    "BiAttention + DCU-LSTM": "discriminative NLI",
    "Biattentive Classification Network + CoVe + Char": "discriminative NLI",
    "CBS-1 + ESIM": "discriminative NLI",
    "Decomposable Attention Model + word2vec": "discriminative NLI",
    "ESIM + ELMo": "discriminative NLI",
    "ESIM + ELMo Ensemble": "discriminative NLI",
    "ESIM + fastText": "discriminative NLI",
    "PairwiseRank +  Multi-Perspective CNN": "discriminative NLI",
    "TLM-I+E": "discriminative NLI",
    "TransNets + SFVerifier + SFEnsembler (ensemble)": "discriminative NLI compound",

    # ── Classical ML classifiers ──────────────────────────────────────────────
    "DV-ngrams-cosine + NB-weighted BON (re-evaluated)": "classical ML",
    "LinearSVM+TFIDF": "classical ML",
    "Logistic Regression (char 2-gram + char 3-gram)": "classical ML",
    "Logistic Regression (word 2-gram + word 3-gram)": "classical ML",
    "Multinomial NB (word 2-gram + word 3-gram)": "classical ML",
    "Random Forest (word 2-gram + word 3-gram)": "classical ML",
    "SVM (word 2-gram + word 3-gram)": "classical ML",
    "SVM + tf-idf (no pre-trained vocab)": "classical ML",
    "SVM + word counts (pre-trained vocab)": "classical ML",
    "SVM with rbf kernel": "classical ML",
    "XGBoost (char 2-gram + char 3-gram)": "classical ML",
    "XGBoost (word 2-gram + word 3-gram)": "classical ML",

    # ── Contrastive / embedding models (not generative) ──────────────────────
    "VSE++ (COCO, ResNet)": "contrastive visual-semantic embedding",
    "VSE++ (COCO, VGG)": "contrastive visual-semantic embedding",
    "VSE++ (Flickr30k, ResNet)": "contrastive visual-semantic embedding",
    "VSE++ (Flickr30k, VGG)": "contrastive visual-semantic embedding",

    # ── MT-specific systems (not general-purpose) ─────────────────────────────
    "Attentional encoder-decoder + BPE": "MT-specific seq2seq",
    "CMLM+LAT+1 iterations": "MT-specific non-autoregressive",
    "CMLM+LAT+4 iterations": "MT-specific non-autoregressive",
    "CSLM + RNN + WP": "speech LM, task-specific",
    "ConvS2S+Risk": "MT-specific convolutional seq2seq",
    "Deep-Att + PosUnk": "MT-specific attention model",
    "DisCo + Mask-Predict (non-autoregressive)": "MT-specific non-autoregressive",
    "Finstreder (Conformer + AMT, character-based)": "ASR compound pipeline",
    "Finstreder (Quartznet + AMT)": "ASR compound pipeline",
    "GNMT+RL": "MT-specific GNMT with RL",
    "ImitKD + Full": "MT-specific imitation learning",
    "NPMT + language model": "MT-specific phrase model",
    "Neural PBMT + LM [Huang2018]": "MT-specific phrase-based",
    "PBSMT + NMT": "MT-specific phrase + neural compound",
    "RNMT+": "MT-specific recurrent model",
    "S2Tree+5gram NPLM": "MT-specific structured model",
    "SMT + iterative backtranslation (unsupervised)": "MT-specific unsupervised",
    "Task Modulation + Multitask Learning(ASR/MT) + Data Augmentation": "MT/ASR compound",
    "Transformer + ASR Pretrain": "MT-specific ASR-pretrained",
    "Transformer + ASR Pretrain + SpecAug": "MT-specific ASR compound",
    "Transformer + Meta Learning(ASR/MT) + Data Augmentation": "MT/ASR compound",
    "Unsupervised NMT + Transformer": "MT-specific unsupervised",
    "Unsupervised NMT + weight-sharing": "MT-specific unsupervised",
    "Unsupervised S2S with attention": "MT-specific unsupervised",
    "Unsupervised attentional encoder-decoder + BPE": "MT-specific unsupervised",
    "mRASP+Fine-Tune": "MT-specific pretrained",

    # ── Knowledge distillation systems (not actual deployed models) ──────────
    "DKD++(T:resnet-32x4, S:resnet-8x4)": "KD system, non-LLM",
    "DKD++(T:resnet50, S:mobilenetv2)": "KD system, non-LLM",
    "KD++(T: ResNet-34 S:ResNet-18)": "KD system, non-LLM",
    "KD++(T: ViT-S, S:resnet18)": "KD system, non-LLM",
    "KD++(T: regnety-16GF S:ViT-B)": "KD system, non-LLM",
    "KD++(T:ViT-B, S:resnet18)": "KD system, non-LLM",
    "KD++(T:renset101 S:resnet18)": "KD system, non-LLM",
    "KD++(T:resnet-152 S:resnet-101)": "KD system, non-LLM",
    "KD++(T:resnet-152 S:resnet-50)": "KD system, non-LLM",
    "KD++(T:resnet-152 S:resnet18)": "KD system, non-LLM",
    "KD++(T:resnet152 S:resnet34)": "KD system, non-LLM",
    "KD++(T:resnet50 S:resnet18)": "KD system, non-LLM",
    "KD++(T:resnet56, S:resnet20)": "KD system, non-LLM",
    "ReviewKD++(T:WRN-40-2, S:WRN-40-1)": "KD system, non-LLM",
    "ReviewKD++(T:resnet-32x4, S:shufflenet-v1)": "KD system, non-LLM",
    "ReviewKD++(T:resnet-32x4, S:shufflenet-v2)": "KD system, non-LLM",
    "ReviewKD++(T:resnet50, S:mobilenet-v1)": "KD system, non-LLM",

    # ── Task-specific discriminative VQA / video / sentiment models ──────────
    "BAN+Glove+Counter": "discriminative VQA",
    "HQI+ResNet": "discriminative VQA",
    "LMH + CSS": "discriminative VQA",
    "LMH + RMFE": "discriminative VQA",
    "LSTM Q+I": "discriminative VQA",
    "LXMERT (Pre-train + scratch)": "discriminative VL encoder",
    "MCAN+VC": "discriminative VQA",
    "MCB+Att.": "discriminative VQA",
    "NMTSRC+IMG": "MT with image compound",
    "STAMP (Sun et al., 2018)+": "aspect sentiment LSTM",
    "TNet-ATT(+AS)": "aspect sentiment network",
    "UNITER + MAC + Graph Networks": "discriminative VQA",
    "UpDn+SCR (VQA-X)": "discriminative VQA",
    "VIOLET + MELTR": "discriminative video-language",
    "X-101 grid features + MCAN": "discriminative VQA",

    # ── Compound retrieval+generation pipelines ───────────────────────────────
    "BART + DPR": "compound: BART + dense retrieval",
    "BART + SimCLS": "compound: BART + reranker",
    "BART+DPR": "compound: BART + dense retrieval",
    "FiD+Distil": "compound: Fusion-in-Decoder + distillation",
    "FiE+PAQ": "compound: entity retrieval + QA",
    "IRRR+": "compound: iterative retrieval + reader",
    "Multitask DPR + BART": "compound: retrieval + generation",
    "PEGASUS + SimCLS": "compound: PEGASUS + reranker",
    "PEGASUS + SummaReranker": "compound: PEGASUS + reranker",
    "Quark + SemanticRetrievalMRS IR": "compound: Quark + retrieval",
    "RETRO + DPR (full)": "compound: RETRO + dense retrieval",

    # ── LLM compound inference or unrecoverable entries ───────────────────────
    "Claude + LATIN-Prompt": "ambiguous Claude version, unrecoverable",
    "DeepMind 70B Model (SFT+ORM-RL, ORM reranking)": "compound: LLM + ORM reranker",
    "DeepMind 70B Model (SFT+PRM-RL, PRM reranking)": "compound: LLM + PRM reranker",
    "GPT-2-Medium 355M + question-solution classifier (BS=1)": "compound: LM + classifier",
    "GPT-2-Medium 355M + question-solution classifier (BS=5)": "compound: LM + classifier",
    "Heinsen Routing + GPT-2": "compound routing architecture, no clean base",
    "LLaMA-2-Chat-7B + Representation Control (Contrast Vector)": "no Llama-2-7B-Chat canonical",
    "LLaMa-2-7B-Chat + TruthX": "no Llama-2-7B-Chat canonical",
    "Mistral multi hop with very large sources": "describes evaluation setup, not a model",
    "SFT-Mistral-7B (Metamath + ovm +ensemble)": "compound: model + ensemble",
    "Shepherd + DeepSeek-67B (SFT on MetaMATH + PRM rerank, k=256)": "compound: LLM + PRM reranker",
    "Shepherd+Mistral-7B (SFT on MetaMATH + PRM RL+ PRM rerank, k=256)": "compound: LLM + PRM reranker",

    # ── Compound NLG / summarization pipelines ────────────────────────────────
    "CodeRL+CodeT5": "compound: CodeRL framework + CodeT5 (no standalone CodeT5 base)",
    "Pointer + Coverage + EntailmentGen + QuestionGen": "compound NLG pipeline",
    "Pointer-Generator + Coverage": "compound: pointer + coverage mechanism (task-specific)",
    "Selector+Pointer Generator": "compound: extractor + abstractor",

    # ── Pre-LLM task-specific models and misc compounds (unclear group) ───────
    "+ Unigram and bigram features": "feature-augmented model",
    "10 LSTM+CNN inputs + SNM10-SKIP (ensemble)": "task-specific ensemble",
    "1200D REGMAPR (Base+Reg)": "task-specific regression model",
    "2-layer skip-LSTM + dropout tuning": "task-specific LSTM",
    "3-Modalities: Unary + Pairwise + Ternary (ResNet)": "task-specific vision model",
    "3D-CNN+LSTM+CE": "task-specific video model",
    "4096D BiLSTM with max-pooling": "discriminative sentence encoder",
    "5TMT-qe+o": "MT quality estimation compound",
    "600D (300+300) BiLSTM encoders": "discriminative sentence encoder",
    "600D (300+300) BiLSTM encoders with intra-attention": "discriminative encoder",
    "600D (300+300) BiLSTM encoders with intra-attention and symbolic preproc.": "discriminative encoder",
    "600D (300+300) Deep Gated Attn. BiLSTM encoders": "discriminative encoder",
    "600D BiLSTM with generalized pooling": "discriminative encoder",
    "ACE + document-context": "task-specific event extractor",
    "AIO+MDF": "video QA model",
    "AIO+MIF": "video QA model",
    "AQQU+TEQUILA": "task-specific KGQA compound",
    "ASP+T5-3B": "task-specific compound (argument structure + T5)",
    "ASP+flan-T5-large": "task-specific compound (argument structure + Flan-T5)",
    "ATRLP+PV (ensemble)": "task-specific ensemble",
    "AVIQA+ (ensemble)": "task-specific video QA ensemble",
    "Abs+": "task-specific abstractive model",
    "AdvAug (aut+adv)": "adversarial augmentation technique",
    "All-in-one+": "task-specific video model",
    "Attentive + relabling + ensemble": "task-specific ensemble",
    "Attentive CNN context with LSTM": "discriminative reader",
    "BCN+Char+CoVe": "discriminative sentence encoder",
    "BCN+ELMo": "discriminative sentence encoder",
    "BCN+Suffix BiLSTM-Tied+CoVe": "discriminative sentence encoder",
    "BM25+CE": "retrieval + cross-encoder compound",
    "BM25-HierSumm (query: step + method + article titles)": "retrieval system",
    "BM25-HierSumm (query: step + method titles)": "retrieval system",
    "BNA + HardDrop (single model)": "task-specific model",
    "BNA + SoftDrop (single model)": "task-specific model",
    "BP-Transformer + GloVe": "task-specific transformer",
    "Balanced+bi-leaf-RNN": "task-specific tree model",
    "Baseline + BS": "task-specific baseline",
    "Bi-GRU (MLE+SLE)": "task-specific GRU",
    "Bi-LSTM + Logic rules": "task-specific LSTM + rules compound",
    "Bi-LSTM+2+5": "task-specific LSTM",
    "Bi-LSTM-CRF + Lexical Features": "discriminative NER model",
    "BiLSTM + MLP (auto syntax)": "discriminative dependency parser",
    "BiLSTM + MLP (gold syntax)": "discriminative dependency parser",
    "BiLSTM + linear-basis-cust": "task-specific model",
    "BiLSTM max-out question-match (WordNet + science fact)": "discriminative QA",
    "BiLSTM max-out question-match (science fact + common knowledge fact)": "discriminative QA",
    "BiLSTM+CHIM": "task-specific model",
    "BiLSTM-CRF+ELMo": "discriminative NER model",
    "BiRNN + GCN (Syn + Sem)": "discriminative model",
    "Bigram-CNN (lexical overlap + dist output)": "task-specific feature model",
    "C2F + ALTERNATE": "task-specific coarse-to-fine model",
    "CES (query: method + article + steps titles)": "retrieval system",
    "CES (query: method + article titles)": "retrieval system",
    "CES (query: step + method + article titles)": "retrieval system",
    "CES (query: step + method titles)": "retrieval system",
    "CIGL-OIE + IGL-CA (OpenIE6)": "information extraction compound",
    "CIGL-OIE + IGL-CA Kolluru et al. (2020)": "information extraction compound",
    "CNN + CRF": "discriminative labeling model",
    "CNN + LSTM + RN": "task-specific model",
    "CNN + LSTM + RN + HAN": "task-specific model",
    "CNN + Logic rules": "task-specific model + rules compound",
    "CNN Large + fine-tune": "task-specific CNN",
    "CNN+GRU+FiLM": "task-specific model",
    "CNN+LSTM": "task-specific model",
    "CNN+Lowercased": "task-specific CNN",
    "CNN+MCFA": "task-specific model",
    "CRF + AutoEncoder": "discriminative labeling model",
    "CVT + Multi-Task": "semi-supervised task-specific model",
    "CVT + Multi-Task + Large": "semi-supervised task-specific model",
    "CoCoNet + CoCoPretrain": "task-specific model",
    "Comp-Clip + LM + LC": "compound visual model",
    "CompassMTL 567M with Tailor": "MT-specific multilingual model",
    "Concept pointer+DS": "task-specific NLG pointer model",
    "Concept pointer+RL": "task-specific NLG compound",
    "Consistency Tree LSTM with tuned Glove vectors [tai2015improved]": "discriminative tree model",
    "ControlCopying + BPNorm": "task-specific NLG model",
    "ControlCopying + SBWR": "task-specific NLG model",
    "Conv-LSTM (deep+pos)": "task-specific Conv-LSTM",
    "Counterfactual Regression + WASS": "task-specific regression",
    "CrossWeigh + Pooled Flair": "discriminative NER model",
    "Cutoff + Relaxed Attention + LM": "task-specific LM compound",
    "Cutoff+Knee": "task-specific model",
    "DCN + Char + CoVe": "discriminative span reader",
    "DGLSTM-CRF + ELMo": "discriminative NER model",
    "DGLSTM-CRF + ELMo (L=2) 3.0pt1-4.51.5": "discriminative NER model",
    "DNC+CUW": "task-specific memory model",
    "DREAM+Unicoder-VL (MSRA)": "discriminative VL compound",
    "DecAtt + DocReader": "discriminative NLI + QA compound",
    "DetIEIMoJIE (ours) + IGL-CA": "information extraction compound",
    "DetIELSOIE + IGL-CA": "information extraction compound",
    "Discnet+RE": "task-specific discourse model",
    "Document LSTM + Document encoding (Choubey et al., 2020)": "task-specific document model",
    "DrQA + seq2seq with copy attention (single model)": "compound reader + generator",
    "Dynamic Entity Repres. + w2v": "task-specific model",
    "EAZI+ (ensemble)": "task-specific ensemble",
    "EnViT5 + MTet": "task-specific MT compound",
    "EndDec+WFE": "task-specific model",
    "EntitySpanFocus+AT (ensemble)": "discriminative NER ensemble",
    "ExtSum + oracle segmentation (extractive)": "task-specific extractive summarizer",
    "ExtSum + supervised segmentation (extractive)": "task-specific extractive summarizer",
    "ExtSum-LG+MMR-Select+": "task-specific extractive summarizer",
    "ExtSum-LG+RdLoss": "task-specific extractive summarizer",
    "External Knowledge With API": "compound model + API",
    "External Knowledge With API + Reranking": "compound model + API + reranker",
    "FrozenBiLM+": "discriminative frozen VL model",
    "GA+MAGE (32)": "task-specific generative model (video/image, not text)",
    "GIT+MDF": "compound vision model + MDF",
    "GRU+Attention": "task-specific GRU",
    "Gated-Attention Reader (+ features)": "discriminative reading comprehension",
    "GloVe + bi-LSTM + CRF": "discriminative NER model",
    "GloVe + bi-LSTM Stanovsky et al. (2018)": "task-specific SRL model",
    "GloVe+Emo2Vec": "discriminative sentiment model",
    "GrapeQA: PEGA+CANP": "compound QA pipeline",
    "GraphIE (GCN+BiLSTM)": "discriminative IE model",
    "GreedyRel (query: method + article + steps titles)": "retrieval system",
    "GreedyRel (query: method + article titles)": "retrieval system",
    "GreedyRel (query: step + method titles)": "retrieval system",
    "Gumbel+bi-leaf-RNN": "task-specific tree model",
    "HSCRF + softdict": "discriminative labeling model",
    "HiStruct+": "task-specific hierarchical summarizer",
    "I-DARTS + Flair": "discriminative NER model",
    "IR++": "task-specific retrieval model",
    "IntNet + BiLSTM-CRF": "discriminative NER compound",
    "JustAsk+": "discriminative video QA",
    "KIGN+Prediction-guide": "task-specific knowledge graph model",
    "LMH+Entropy regularization": "discriminative VQA bias model",
    "LMH+Entropy regularization (Ensemble)": "discriminative VQA bias ensemble",
    "LOGNet+VLR": "task-specific video grounding",
    "LSTM (lexical overlap + dist output)": "discriminative feature model",
    "LSTM + global features": "task-specific LSTM",
    "LSTM with dynamic skip": "task-specific LSTM",
    "LSTM+Attention+Ensemble": "task-specific ensemble",
    "LSTM+CNN": "task-specific model",
    "LSTM+SynATT+TarRep": "aspect-based sentiment LSTM",
    "LSTM-8192-1024 + CNN Input": "task-specific LSTM",
    "LSTM6 + PosUnk": "MT-specific LSTM",
    "LUKE + SubRegWeigh (K-means)": "task-specific NER compound",
    "Learned-Mixin +H": "discriminative VQA bias model",
    "LexRank (query: method + article + steps titles)": "retrieval system",
    "LexRank (query: method + article titles)": "retrieval system",
    "LexRank (query: step + method + article titles)": "retrieval system",
    "LexRank (query: step + method titles)": "retrieval system",
    "MAT+Knee": "task-specific model",
    "MGM-7B+RP": "task-specific model compound",
    "MHPGM + NOIC": "discriminative VQA model",
    "ML + Intra-Attention (Paulus et al., 2017)": "task-specific seq2seq (summarization only)",
    "ML + RL (Paulus et al., 2017)": "task-specific seq2seq (summarization only)",
    "ML+RL ROUGE+Novel, with LM": "task-specific summarization compound",
    "ML+RL, with intra-attention": "task-specific summarization compound",
    "MLM+ del-word+ reorder": "task-specific data-augmented model",
    "MRN + global features": "task-specific model",
    "MSR + MS Cog. Svcs.": "compound ASR pipeline",
    "MSR + MS Cog. Svcs., X10 models": "compound ASR ensemble",
    "Multi-task BiLSTM + Attn": "discriminative sentiment model",
    "NAT +FT + NPD": "MT-specific non-autoregressive",
    "NCRF++": "discriminative NER model",
    "NMN+LSTM+FT": "discriminative VQA compound",
    "Neo-6B (QA + WS)": "task-specific fine-tuned model (no canonical Neo-6B base)",
    "Neural-CRF+AE": "discriminative labeling model",
    "NeuralCRF+SAC": "discriminative labeling model",
    "Noise-robust Co-regularization + LUKE": "task-specific NER compound",
    "NormTab (Targeted) + SQL": "compound table QA + SQL",
    "OTF dict+spelling (single)": "task-specific OCR/MT model",
    "OTF spelling+lemma (single)": "task-specific OCR/MT model",
    "OpenIE6 (CIGL-OIE + IGL-CA)": "information extraction compound",
    "PEBG+DKT": "task-specific educational model",
    "PEBG+DKVMN": "task-specific educational model",
    "PEVL+": "task-specific VL model",
    "PRET+MULT": "task-specific pretraining method",
    "PTGEN + Coverage": "task-specific summarization compound",
    "PaLI (ft SNLI-VE + Synthetic Data)": "task-specific fine-tuned VL model (SNLI-VE only)",
    "Paragraph vector (lexical overlap + dist output)": "discriminative feature model",
    "Primal.+Trans.": "task-specific LM method",
    "Pythia v0.3 + LoRRA": "task-specific VQA compound",
    "QANet + data augmentation ×3": "discriminative span reader",
    "QUINT+TEQUILA": "task-specific KGQA compound",
    "RGAT+": "task-specific aspect sentiment model",
    "RL + pg + cbdec": "task-specific summarization compound",
    "RNN-1024 + 9 Gram": "task-specific LSTM+LM",
    "ROUGESal+Ent RL": "task-specific summarization compound",
    "RQA+IDR (single model)": "discriminative reader",
    "ReVeaL WIT + CC12M + Wikidata + VQA-2": "compound VQA + retrieval",
    "Real + synthetic": "task-specific augmented training",
    "SAINT+": "task-specific tabular model",
    "SIP + BCAUSS": "task-specific model",
    "SMT+LSTM5": "MT-specific compound",
    "STAMP+RL (Sun et al., 2018)+": "aspect sentiment RL model",
    "STM+TSED+PT+2L": "task-specific temporal model",
    "Scrambled code + broken": "task-specific code baseline",
    "Scrambled code + broken (alter)": "task-specific code baseline",
    "Seq-KD + Seq-Inter + Word-KD": "MT-specific KD compound",
    "Seq2CNN with GWS(50)": "task-specific CNN decoder",
    "Seq2Seq + Attention": "MT-specific seq2seq",
    "Seq2seq + E2T_cnn": "task-specific compound",
    "Seq2seq + selective + MTL + ERAM": "task-specific summarization compound",
    "SetFit + OCD": "task-specific sentence encoder + classifier",
    "SetFit + OCD(5)": "task-specific sentence encoder + classifier",
    "SpanModel + SequenceLabelingModel": "discriminative compound",
    "Stack 4-layer RNNSearch + Dual Learning + Deliberation Network": "MT-specific compound",
    "Struct+2Way+Word": "task-specific model",
    "T5(Tan and Bansal, 2019) + Prefixes": "task-specific T5 variant",
    "TabSQLify (col+row)": "compound table QA + SQL",
    "Tall Transformer with Style-Augmented Training": "task-specific model",
    "TbD + reg + hres": "discriminative VQA compound",
    "Transformer (big) + Relative Position Representations": "MT-specific transformer",
    "Transformer + R-Drop": "task-specific (MT/NLG, insufficient base info to remap)",
    "Transformer + R-Drop + Cutoff": "task-specific compound",
    "Transformer + SRU": "task-specific hybrid",
    "Transformer Big + MoS": "MT-specific transformer",
    "Transformer Big + adversarial MLE": "MT-specific transformer",
    "Transformer Big with FRAGE": "MT-specific transformer",
    "Transformer Multitask + LayerDrop": "MT-specific compound",
    "Transformer with Adapters": "task-specific (unclear base)",
    "Transformer with FRAGE": "task-specific LM (no canonical base, single metric)",
    "Transformer+BPE+FixNorm+ScaleNorm": "MT-specific transformer",
    "Transformer+BPE-dropout": "MT-specific transformer",
    "Transformer+BT (ADMIN init)": "MT-specific transformer",
    "Transformer+LRPE+PE+ALONE+Re-ranking": "MT-specific compound with reranking",
    "Transformer+LRPE+PE+Re-ranking+Ensemble": "MT-specific ensemble",
    "Transformer+LayerNorm-simple": "MT-specific transformer",
    "Transformer+Rep(Sim)+WDrop": "task-specific (MT, WDrop=word dropout)",
    "Transformer+Rep(Uni)": "task-specific (MT+summarization, no canonical base)",
    "Transformer+WDrop": "task-specific (summarization/MT, word dropout)",
    "Translate-Source + fastText": "MT compound",
    "Translate-Target + fastText": "MT compound",
    "Two-Stage + RL": "task-specific summarization compound",
    "TypeSQL+TC (Yu et al., 2018)+": "task-specific text-to-SQL",
    "USE_T+CNN": "discriminative text classifier",
    "USE_T+CNN (lrn w.e.)": "discriminative text classifier",
    "USE_T+CNN (w2v w.e.)": "discriminative text classifier",
    "USSM + Cause-Effect Knowledge Base": "task-specific causal model",
    "USSM + Supervised Deepnet": "task-specific model",
    "USSM + Supervised Deepnet + 3 Knowledge Bases": "task-specific compound",
    "UniMD+Sync.": "task-specific video temporal grounding",
    "VIOLET+": "discriminative video-language model",
    "VS^3-NET (single model)": "task-specific model",
    "VinVL+L": "discriminative VL model",
    "Word+ES (Scratch)": "task-specific word embedding model",
    "Word-level CNN+LSTM (full scoring)": "discriminative code model",
    "Word-level CNN+LSTM (partial scoring)": "discriminative code model",
    "Zeropoint LLM.int8 13B (vector-wise + decomp)": "quantization method entry, unknown base",
    "araneum fasttext + GRU": "task-specific Russian NLP",
    "araneum fasttext + LSTM": "task-specific Russian NLP",
    "araneum word2vec (skipgram) + GRU": "task-specific Russian NLP",
    "araneum word2vec (skipgram) + LSTM": "task-specific Russian NLP",
    "attention+self-attention (single model)": "task-specific attention model",
    "d-LSTM+nI": "task-specific LSTM",
    "del+obj": "task-specific model",
    "oh-CNN + two LSTM tv-embed.": "task-specific model",
    "pGSLM+": "spoken language model (not text generative)",
    "r-net+ (ensemble)": "discriminative reader",
    "r-net+ (single model)": "discriminative reader",
    "rnn-ext + RL": "task-specific extractive summarizer",
    "rnn-ext + abs + RL + rerank": "compound extractive+abstractive+reranker",
    "ruscorpora fasttext + GRU": "task-specific Russian NLP",
    "ruscorpora fasttext + LSTM": "task-specific Russian NLP",
    "ruscorpora word2vec (skipgram) + GRU": "task-specific Russian NLP",
    "sMIM (1024) +": "task-specific model",
    "single-hop + LCGN (ours)": "task-specific VQA compound",
    "CodeChain + WizardCoder-15B": "compound: CodeChain framework + WizardCoder LLM",
}

# ─────────────────────────────────────────────────────────────────────────────
# RENAME — create canonical base from compound name (no existing canonical)
# Applied FIRST so SETUP_REMAP targets exist.
# ─────────────────────────────────────────────────────────────────────────────
RENAME = {
    "GPT-Neo 125M + Self-Sampling": "GPT-Neo 125M",   # 1r → base
    "CodeGen-Mono 16B + CodeT":    "CodeGen-Mono 16B", # 1r → base
    "code-davinci-001 175B + CodeT": "code-davinci-001 175B", # 1r → base
}

RENAME_SETUP = {
    # old_id (post-rename new_id): setup to write on newly-renamed result rows
    "GPT-Neo 125M":       "Self-Sampling",
    "CodeGen-Mono 16B":   "CodeT",
    "code-davinci-001 175B": "CodeT",
}

# ─────────────────────────────────────────────────────────────────────────────
# REMAP — merge into existing canonical; delete old entry.
# ─────────────────────────────────────────────────────────────────────────────
REMAP = {
    # ── LLaMA 3 8B + MoSLoRA aliases → canonical ─────────────────────────────
    "LLaMA 3 8B+MoSLoRA":  "LLaMA 3 8B + MoSLoRA",  # 2r
    "LLaMA-3 8B+MoSLoRA":  "LLaMA 3 8B + MoSLoRA",  # 1r
    "LLaMA3 8B+MoSLoRA":   "LLaMA 3 8B + MoSLoRA",  # 2r
    "LLaMA3+MoSLoRA":      "LLaMA 3 8B + MoSLoRA",  # 1r (size implicit from benchmark context)
}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP_REMAP — old_id → (base_id, setup_val)
# Applied AFTER RENAME so new bases exist.
# ─────────────────────────────────────────────────────────────────────────────
SETUP_REMAP = {
    # ── Inference-time intervention / representation engineering ─────────────
    "Alpaca 7B + Inference Time Intervention (ITI)": ("Alpaca-7B",          "ITI"),
    "LLaMA 7B + Inference Time Intervention (ITI)":  ("LLaMA 7B",           "ITI"),
    "Vicuna 7B + Inference Time Intervention (ITI)":  ("Vicuna-7B",          "ITI"),
    "LLaMA-2-Chat-13B + Representation Control (Contrast Vector)":
        ("Llama-2-13B-Chat", "Representation Control (Contrast Vector)"),
    "Mistral-7B-Instruct-v0.2 + TruthX": ("Mistral-7B-Instruct-v0.2", "TruthX"),

    # ── Classifier-free guidance ─────────────────────────────────────────────
    "LLaMA 7B + CFG":   ("LLaMA 7B",    "CFG"),
    "LLaMA 13B + CFG":  ("LLaMA 13B",   "CFG"),
    "LLaMA-13B+CFG":    ("LLaMA 13B",   "CFG"),
    "LLaMA 30B + CFG":  ("LLaMA (30B)", "CFG"),
    "LLaMA-30B+CFG":    ("LLaMA (30B)", "CFG"),
    "LLaMA 65B + CFG":  ("LLaMA 65B",   "CFG"),
    "LLaMA-65B+CFG":    ("LLaMA 65B",   "CFG"),

    # ── Post-training alignment methods (CAPO, LATIN-Prompt) ─────────────────
    "Llama-3.3-70B + CAPO":      ("LLaMA-3.3-70B",                  "CAPO"),
    "Mistral-Small-24B + CAPO":  ("Mistral-Small-24B-Instruct-2501", "CAPO"),
    "Qwen2.5-32B + CAPO":        ("Qwen2.5-32B",                    "CAPO"),
    "GPT-3.5 + LATIN-Prompt":    ("GPT-3.5-Turbo",                  "LATIN-Prompt"),
    "GPT-4o +text rationale +IoT": ("GPT-4o",                       "text rationale + IoT"),

    # ── Self-Sampling / CC ────────────────────────────────────────────────────
    "GPT-Neo-2.7B + Self-Sampling": ("GPT-Neo (2.7 B)", "Self-Sampling"),
    "GPT-J + CC":                   ("GPT-J (6B)",      "CC"),

    # ── Code generation post-processing (CodeT, MBR-Exec, REPLUG, LEVER) ─────
    "code-davinci-002 175B + CodeT":         ("code-davinci-002 175B", "CodeT"),
    "code-davinci-002 175B + Coder-Reviewer": ("code-davinci-002 175B", "Coder-Reviewer"),
    "code-davinci-002 175B + LEVER":         ("code-davinci-002 175B", "LEVER"),
    "code-davinci-002 175B + MBR-Exec":      ("code-davinci-002 175B", "MBR-Exec"),
    "code-davinci-002 175B + REPLUG":        ("code-davinci-002 175B", "REPLUG"),
    "code-davinci-002 175B + REPLUG LSR":    ("code-davinci-002 175B", "REPLUG LSR"),
    "code-davinci-002 175B + Reviewer":      ("code-davinci-002 175B", "Reviewer"),
    "code-davinci-001 175B + MBR-Exec":      ("code-davinci-001 175B", "MBR-Exec"),  # base created by RENAME
    "code-cushman-001 12B + MBR-Exec":       ("code-cushman-001 (12B)", "MBR-Exec"),
    "InCoder 6.7B + CodeT":         ("InCoder 6.7B", "CodeT"),
    "InCoder 6.7B + Coder-Reviewer": ("InCoder 6.7B", "Coder-Reviewer"),
    "InCoder 6.7B + MBR-Exec":      ("InCoder 6.7B", "MBR-Exec"),
    "InCoder 6.7B + Reviewer":      ("InCoder 6.7B", "Reviewer"),
    "CodeGen 16B + Coder-Reviewer":  ("CodeGen-16B-multi", "Coder-Reviewer"),
    "CodeGen 16B + MBR-Exec":        ("CodeGen-16B-multi", "MBR-Exec"),
    "CodeGen 16B + Reviewer":        ("CodeGen-16B-multi", "Reviewer"),

    # ── Test-time fine-tuning (SIFT) ─────────────────────────────────────────
    "Test-Time Fine-Tuning with SIFT + GPT-2 (124M)":  ("GPT-2",              "SIFT TTFT"),
    "Test-Time Fine-Tuning with SIFT + GPT-2 (774M)":  ("GPT-2 Large (774M)", "SIFT TTFT"),
    "Test-Time Fine-Tuning with SIFT + Llama-3.2 (1B)": ("Llama-3.2-1B",     "SIFT TTFT"),
    "Test-Time Fine-Tuning with SIFT + Llama-3.2 (3B)": ("Llama-3.2-3B",     "SIFT TTFT"),
    "Test-Time Fine-Tuning with SIFT + Phi-3 (3.8B)":  ("Phi-3 3.8B",        "SIFT TTFT"),

    # ── BART training-technique variants ─────────────────────────────────────
    "BART + R-Drop":   ("BART-Large", "R-Drop"),
    "BART+R3F":        ("BART-Large", "R3F"),
    "BART + Adapters + Lohfink-Rossi-Leaveout (single-model)": ("BART-Large", "Adapters + LRL"),
    "BART-large (s=abs+disci, t=TLDR)": ("BART-Large", "abs+disci pretraining, TLDR FT"),
    "BART-large (s=abs+title, t=TLDR)": ("BART-Large", "abs+title pretraining, TLDR FT"),

    # ── PEGASUS training-technique variants ───────────────────────────────────
    "PEGASUS 2B + SLiC": ("PEGASUS", "SLiC"),
    "Pegasus+DotProd":   ("PEGASUS", "DotProd attention"),

    # ── MatCha with dataset ───────────────────────────────────────────────────
    "MatCha4096 + LaMenDa": ("MatCha", "LaMenDa data, 4096 ctx"),

    # ── Multimodal CoT methods ────────────────────────────────────────────────
    "InstructBLIP-CCoT":    ("InstructBLIP", "CCoT"),
    "OpenFlamingo + CCoT":  ("OpenFlamingo", "CCoT"),
    "OpenFlamingo + DDCoT": ("OpenFlamingo", "DDCoT"),

    # ── Prompting format ──────────────────────────────────────────────────────
    "ChatGPT 3.5 with LAPDoc Prompt (SpatialFormat)": ("GPT-3.5-Turbo", "LAPDoc Prompt (SpatialFormat)"),

    # ── BLIP with graph-text augmentation ────────────────────────────────────
    "BLIP (+Graph Text)":            ("BLIP", "Graph Text"),
    "BLIP (+Graph Text, +Graph Neg)": ("BLIP", "Graph Text + Graph Neg"),
}


def validate(m_df):
    existing = set(m_df["model_id"])
    will_exist = existing | set(RENAME.values())
    ok = True
    for old_id in REMOVE:
        if old_id not in existing:
            print(f"  WARN  REMOVE source missing: {old_id!r}")
    for old_id, new_id in RENAME.items():
        if old_id not in existing:
            print(f"  WARN  RENAME source missing: {old_id!r}")
        if new_id in existing:
            print(f"  ERROR RENAME target already exists (use REMAP): {new_id!r}")
            ok = False
    for old_id, new_id in REMAP.items():
        if old_id not in existing:
            print(f"  WARN  REMAP source missing: {old_id!r}")
        if new_id not in existing:
            print(f"  ERROR REMAP target missing: {new_id!r}")
            ok = False
    for old_id, (base_id, _) in SETUP_REMAP.items():
        if old_id not in existing:
            print(f"  WARN  SETUP_REMAP source missing: {old_id!r}")
        if base_id not in will_exist:
            print(f"  ERROR SETUP_REMAP base missing: {base_id!r}")
            ok = False
    return ok


def apply(m_df, r_df, write: bool):
    m = m_df.copy()
    r = r_df.copy()

    # ── 1. REMOVE ─────────────────────────────────────────────────────────────
    n_removed_models = 0
    n_removed_results = 0
    for old_id, reason in REMOVE.items():
        cur = set(m["model_id"]) if write else set(m_df["model_id"])
        if old_id not in cur:
            print(f"  WARN  REMOVE source missing: {old_id!r}")
            continue
        n = (r_df["model_name"] == old_id).sum()
        print(f"  REMOVE  {old_id!r}  ({n}r)  [{reason}]")
        if write:
            m = m[m["model_id"] != old_id]
            r = r[r["model_name"] != old_id]
            n_removed_models += 1
            n_removed_results += n

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

    # Set setup for RENAME entries (they had setup qualifier in old name)
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
        cur = set(m["model_id"]) if write else set(m_df["model_id"])
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

    return m, r, n_removed_models, n_removed_results, dupes_dropped


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
    m_new, r_new, n_rm, n_rr, n_dupes = apply(m_df, r_df, write=args.write)

    if args.write:
        io.save_csv(m_new, config.MODELS_CSV)
        io.save_results(r_new)
        print(f"\nWritten.")
        print(f"  models:  {len(m_df)} → {len(m_new)}  (−{len(m_df)-len(m_new)}, "
              f"{n_rm} removed)")
        print(f"  results: {len(r_df)} → {len(r_new)}  (−{len(r_df)-len(r_new)}, "
              f"{n_rr} directly removed, {n_dupes} post-merge dupes dropped)")
        print("\nNext steps:")
        print("  python3 scripts/verify_data.py")
        print("  python3 scripts/manage_data.py recompute-stats --write")
    else:
        n_remove = len(REMOVE)
        n_rename = len(RENAME)
        n_remap = len(REMAP)
        n_setup = len(SETUP_REMAP)
        print(f"\nDry run summary:")
        print(f"  REMOVE: {n_remove}  RENAME: {n_rename}  REMAP: {n_remap}  "
              f"SETUP_REMAP: {n_setup}")
        print(f"\nPass --write to apply.")


if __name__ == "__main__":
    sys.exit(main())
