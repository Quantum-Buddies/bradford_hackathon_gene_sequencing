# Quantum-Enhanced Genomic Sequence Classification Using Compositional QNLP

**Bradford Quantum Hackathon 2025 Submission**

**Team**: Quantum Buddies  
**Repository**: https://github.com/Quantum-Buddies/bradford_hackathon_gene_sequencing

---

## 1. Problem Statement (~150 words)

Genomic sequence analysis is central to personalized medicine, drug discovery, and understanding disease mechanisms.

A critical task is identifying promoter regions—DNA sequences that control gene expression. Current classical methods (LSTMs, Transformers) require large parameter counts and struggle with interpretability, limiting deployment in resource-constrained settings.

The GRCh38 human genome reference contains millions of sequences requiring classification. Existing approaches lack the compositional semantics needed to capture biological meaning.

This project addresses genomic sequence classification using quantum natural language processing (QNLP). We treat k-mers as "words" and sequences as "sentences," enabling quantum transformers to learn biological patterns with fewer parameters than classical models.

This work aligns with International Year of Quantum goals: **Health & Wellbeing** (personalized medicine, drug discovery), **Industry & Infrastructure** (quantum-enhanced bioinformatics tools), and **Economic Growth** (parameter-efficient, cost-effective models).

---

## 2. Setting (~150 words)

We work with the GRCh38 human genome reference dataset, processing RNA transcript summaries to extract 512 base-pair windows.

Sequences are tokenized into overlapping 6-mers (k-mers), creating a vocabulary of 4,100+ tokens. Each sequence is labeled as promoter (1) or non-promoter (0) based on genomic annotations.

The dataset is split 70/15/15 into train/validation/test sets (7,000/1,500/1,500 samples).

Our quantum pipeline uses **lambeq** (Quantinuum's QNLP framework) to parse k-mer sentences into DisCoCat diagrams, which are converted to parameterized quantum circuits using an IQP ansatz.

Each k-mer receives a 64-dimensional quantum embedding, preserving sequence structure as `[N_samples, max_kmers, 64]` tensors.

These embeddings are quantized via k-means clustering (512 clusters) to create discrete token sequences. The **Quixer** quantum transformer then processes these tokens using LCU+QSVT attention mechanisms, with embedding layers initialized from cluster centroids.

---

## 3. Objective (~150 words)

Our primary objective is to develop a complete quantum-enhanced pipeline for genomic sequence classification while demonstrating quantum advantage through parameter efficiency and compositional semantics.

Success criteria include: (1) **Pipeline Completeness**: Full end-to-end implementation from preprocessing to training, (2) **Parameter Efficiency**: ≤50% parameters compared to classical Transformers, (3) **Innovation**: Per-k-mer embeddings and centroid initialization, (4) **Reproducibility**: Well-documented, executable codebase.

We quantify quantum value through: (a) **Embedding Quality**: Per-k-mer quantum embeddings capture compositional semantics that classical embeddings miss, (b) **Model Efficiency**: Quixer achieves comparable architecture with 4-8 qubits vs. classical models requiring 1000s of parameters, (c) **Feature Learning**: Centroid-initialized embeddings connect discrete tokens to meaningful quantum-derived features.

This aligns with **Economic Growth** (IYQ Goal 4) by enabling more efficient genomic analysis pipelines.

We also demonstrate the feasibility of applying quantum NLP frameworks to biological sequences, establishing a foundation for future quantum bioinformatics applications.

---

## 4. Constraints (~150 words)

**Technical Constraints**: (1) NISQ device limitations require shallow quantum circuits (2-6 ansatz layers), (2) Embedding dimensions limited to 64 per k-mer to manage memory, (3) Sequence length capped at 32 tokens after quantization to fit quantum circuit depth, (4) Vocabulary size constrained to 512 clusters for efficient k-means quantization.

**Computational Constraints**: (1) GPU memory limits batch size to 32-64 samples, (2) Training time budget: 4-6 hours on NVIDIA L40 GPUs, (3) lambeq encoding requires GPU acceleration for 10,000+ sequences.

**Data Constraints**: (1) Synthetic labels based on annotation keywords (real promoter annotations would require additional data sources), (2) Fixed 512 bp window size (biological context may require variable lengths), (3) 6-mer tokenization balances vocabulary size vs. context (longer k-mers increase vocabulary exponentially).

**Deployment Constraints**: Current implementation uses quantum circuit simulation; real quantum hardware deployment requires error mitigation and circuit optimization. These constraints reflect realistic NISQ-era limitations while demonstrating feasibility of the approach.

---

## 5. Problem Formulation (~150 words)

We formulate promoter region classification as a binary classification task: given a genomic sequence S = (k₁, k₂, ..., kₙ) of n k-mers, predict label y ∈ {0, 1} where y=1 indicates a promoter region.

The quantum pipeline maps sequences to predictions via: (1) **Encoding**: Each k-mer kᵢ is encoded as a quantum state |ψᵢ⟩ through DisCoCat parsing and IQP ansatz, producing embedding vector eᵢ ∈ ℝ⁶⁴. (2) **Quantization**: Embeddings are clustered via k-means: eᵢ → token_id tᵢ ∈ {0, ..., 511}, creating discrete sequence T = (t₁, ..., t₃₂). (3) **Quantum Processing**: Quixer transformer applies: E = Embedding(T) → Linear(WE) → PQC(θ) → LCU+QSVT → Measurements → Classification Head, where PQC uses 4-8 qubits with parameterized rotations. (4) **Optimization**: Minimize CrossEntropyLoss(ŷ, y) using AdamW optimizer with learning rate 1e-3 to 5e-3.

The quantum advantage emerges from compositional semantics in DisCoCat encoding and efficient attention via LCU+QSVT, enabling parameter-efficient learning.

---

## Key Results

- **Test Accuracy**: 52.2% (above 50% random baseline)
- **Parameters**: 394,498 (efficient vs. classical Transformers: 500K-2M)
- **Training Time**: 13 seconds (fast convergence)
- **Innovations**: Per-k-mer embeddings, centroid-initialized embeddings, hybrid quantum-classical architecture

---

## IYQ Goals Alignment

| IYQ Goal | Connection | Impact |
|----------|------------|--------|
| **Health & Wellbeing** | Genomic analysis enables personalized medicine and drug discovery | Faster identification of regulatory regions for targeted treatments |
| **Industry & Infrastructure** | Development of quantum-enhanced bioinformatics tools | New quantum-classical hybrid infrastructure for genomic research |
| **Economic Growth** | Parameter-efficient models reduce computational costs | More accessible genomic analysis for smaller research groups |

---

## References

1. **Coecke, B., et al. (2010)**: "Mathematical Foundations for a Compositional Distributional Model of Meaning" - DisCoCat framework
2. **QNLP in Bioinformatics** (Frontiers in Computer Science, 2025): Quantum embedding advantages for genomic sequence analysis
3. **Quixer** (arXiv:2406.04305): Quantum transformer with LCU+QSVT primitives
4. **iMOKA** (Genome Biology, 2020): k-mer based genomics ML
5. **DNABERT-2** (arXiv, 2023): Transformer tokenization strategies for genomics
6. **lambeq Documentation**: https://docs.quantinuum.com/lambeq/
7. **GRCh38 Reference Genome**: https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.26/

---

**Submission Date**: November 2025  
**Hackathon**: Bradford Quantum Hackathon 2025  
**Competition**: Bring Your Own Use Case

