# Residual Connections Are Not Required for 1D Bacterial Raman Spectra: A Controlled Clinical Benchmark

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/overleaf/main.pdf)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IEEE Format](https://img.shields.io/badge/Format-IEEE%20Transactions-blueviolet.svg)](paper/overleaf/main.tex)

Official codebase and publication repository for:
> **"Residual Connections Are Not Required for 1D Bacterial Raman Spectra: A Controlled Clinical Benchmark"**  
> *Preprint / Manuscript available in [`paper/overleaf/main.pdf`](paper/overleaf/main.pdf)*

---

## 🔬 Overview & Key Findings

Deep residual networks (ResNet) have become the de-facto standard across 2D computer vision and have been widely adopted for 1D spectroscopic classification. However, 1D vibrational spectra possess fundamentally distinct inductive biases: unlike 2D natural images characterized by spatial hierarchies and translation invariance, 1D Raman spectra consist of sharp, localized molecular vibrational peaks superimposed on slowly varying fluorescence backgrounds.

In this work, we conduct a strictly controlled empirical benchmark comparing **1D Plain Convolutional Neural Networks (Plain CNN)** against **1D Residual Networks (1D ResNet)** on the public 30-isolate bacterial Raman benchmark (60,000 reference spectra, 3,000 test spectra; Ho et al., *Nature Communications* 2019).

### 🏆 Benchmark Highlights

| Metric / Task | 1D Plain CNN (Ours) | 1D ResNet (Matched) | Reference ResNet-25 (Ho et al.) |
| :--- | :---: | :---: | :---: |
| **Parameters** | **40,446** (1×) | 262,510 (6.5×) | 1,340,878 (33.1×) |
| **Depth** | **6 layers** | 17 layers | 25 layers |
| **30-Isolate Test Accuracy** | **83.9% ± 0.4%** | 79.6% ± 0.6% | 82.2% |
| **Macro F1-Score** | **0.835** | 0.791 | — |
| **8-Class Empiric Treatment Acc.** | **96.5%** | 94.8% | — |
| **MRSA vs. MSSA ROC-AUC** | **0.941** | 0.924 | — |
| **Noise Robustness ($\mathrm{SNR}=1$)** | **31.3%** | 24.1% | — |
| **Fine-Tuning Domain Gain** | **+3.9%** | +2.7% | — |

### 💡 Core Insights
1. **Residual Shortcut Penalties**: In 1D low-SNR spectroscopy, deep residual identity shortcuts add excess network capacity that memorizes high-frequency stochastic sensor noise rather than underlying chemical bond vibrations.
2. **Receptive Field Advantage**: The Plain CNN achieves an effective receptive field of **971 spectral points** (vs. 435 points for ResNet), enabling the model to learn long-range cross-band correlations across the entire fingerprint region ($382$--$1792\text{ cm}^{-1}$).
3. **Pre-processing Ablation (Appendix A)**: Classical chemometric pre-processing pipelines (Whitaker--Hayes despiking, Savitzky--Golay polynomial smoothing, and ASPLS baseline correction) **reduce test accuracy from 85.3% down to 76.5%**. 1D convolutional layers act as adaptive finite-impulse response (FIR) filters; polynomial smoothing strips the narrow diagnostic vibrational shoulders that distinguish bacterial strains.

---

## 📁 Repository Structure

```
.
├── paper/
│   ├── overleaf/              # Complete 14-page IEEE manuscript
│   │   ├── main.tex           # LaTeX source code
│   │   ├── main.pdf           # Compiled publication PDF (592 KB)
│   │   ├── references.bib     # 27+ fully verified BibTeX entries
│   │   ├── figures/           # 9 publication-grade vector PDFs (no raster images)
│   │   └── word-docs/         # Co-author original drafts & source text
│   ├── scripts/               # Publication figure generation scripts
│   │   ├── make_fig_spectra.py           # Fig. 1: Mean spectra with characteristic bands
│   │   ├── generate_missing_figures.py   # Fig. 4 & 8: Confusion matrices & loss curves
│   │   ├── generate_treatment_heatmaps.py# Fig. 5: 8-class empiric treatment heatmaps
│   │   └── generate_vector_figures.py    # Fig. 6 & 7: MRSA ROC curves & SNR degradation
│   └── README.md              # Instructions for Overleaf compilation & figure reproduction
│
├── src/                       # Core modeling and evaluation framework
│   ├── model.py               # Plain CNN, 1D ResNet, Transformer, and LSTM implementations
│   ├── dataset.py             # RamanDataset loader with on-the-fly augmentation & caching
│   ├── engine.py              # Training loop with macro-F1, confusion matrix, & stats
│   ├── repro.py               # Deterministic seed fixing (CPU, GPU, MPS)
│   └── tools/                 # Ablation and Phase-2 experiment runners
│
├── legacy_baseline/           # 6-layer Plain CNN architecture reference
├── foundation/                # Foundation model pretraining & adaptation module (corpus v1, LoRA)
├── docs/                      # Scientific documentation & experiment logs
│   ├── 11_PREPROCESSING_IMPACT.md           # Pre-processing ablation analysis (Appendix A)
│   ├── 12_ANTIBIOTIC_TREATMENT_GROUPINGS.md # 30-to-8 clinical treatment class mapping
│   ├── 13_TREATMENT_GROUP_RESULTS.md        # Empiric grouping accuracy breakdown
│   ├── foundation_model/      # Foundation model design specs & execution plans
│   ├── rnn/                   # Exploratory RNN/LSTM research notes
│   ├── sprints/               # Historical sprint logs & pre-processing visualizations
│   └── archive/               # Project archive notes & slide decks
│
├── scripts/                   # Repository utility scripts
│   ├── evaluate_changes.py    # Pipeline change evaluator
│   ├── visualize_preprocessing.py # Preprocessing step-by-step plotter
│   └── convert_docs_to_pdf.py # Markdown-to-PDF converter
│
├── literature/                # Reference literature and survey papers
├── results/                   # Experimental logs, confusion matrices, and ablation data
├── runs/                      # Reproducibility run manifests and per-epoch metrics
└── tests/                     # Unit and integration test suite
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/hspgit/RAMAN-Effect.git
cd RAMAN-Effect

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the 30-isolate bacterial Raman dataset from [Ho et al. (2019)](https://doi.org/10.1038/s41467-019-12898-9) and place the numpy arrays in `data/`:
```bash
data/
├── X_reference.npy  # (60,000, 1000)
├── y_reference.npy  # (60,000,)
├── X_finetune.npy   # (3,000, 1000)
├── y_finetune.npy   # (3,000,)
├── X_test.npy       # (3,000, 1000)
└── y_test.npy       # (3,000,)
```

### 3. Training the Models

Train the 6-layer Plain CNN (Legacy Baseline) end-to-end:
```bash
python main.py --model-type legacy_cnn --pipeline minmax_only --epochs 100 --finetune-epochs 20 --seeds 1,2,3
```

Train the matched 1D ResNet:
```bash
python main.py --model-type resnet --pipeline minmax_only --epochs 100 --finetune-epochs 20 --seeds 1,2,3
```

### 4. Reproducing Paper Figures (100% Vector PDF)

All figures in the paper can be regenerated from experimental checkpoints with:

```bash
# Fig. 1: Mean spectra with characteristic vibrational bands
python paper/scripts/make_fig_spectra.py

# Fig. 4 & 8: 30-class confusion matrices and training loss curves
python paper/scripts/generate_missing_figures.py

# Fig. 5: 8-class empiric antibiotic treatment heatmaps
python paper/scripts/generate_treatment_heatmaps.py

# Fig. 6 & 7: MRSA vs. MSSA ROC curves, noise decay, and fine-tuning domain gain
python paper/scripts/generate_vector_figures.py
```
*Generated vector PDFs are written directly to [`paper/overleaf/figures/`](paper/overleaf/figures).*

### 5. Compiling the Paper Locally

```bash
cd paper/overleaf
latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex
```

---

## 📖 Citation

If you find this work, benchmark, or codebase useful, please cite:

```bibtex
@article{raman_plain_cnn_benchmark2026,
  title   = {Residual Connections Are Not Required for 1D Bacterial {Raman} Spectra: A Controlled Clinical Benchmark},
  author  = {Pradhan, Hrishikesh Suresh and Jaimes, Amos and Siddhawar, K. and Olano, Edward M. and Brown, Nik Bear},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## 🏛️ Project Background & Broader Context

*(Preserved from the foundational public health surveillance project)*

### Public Health Surveillance through AI-Driven Spectroscopy

Public health surveillance faces unprecedented challenges in the 21st century, from emerging infectious diseases to antimicrobial resistance and substance abuse monitoring. Traditional surveillance methods often suffer from limitations in reach, timeliness, and cost-effectiveness. The RAMAN Effect project integrates cutting-edge technologies: Surface-Enhanced Raman Spectroscopy (SERS), Wastewater-Based Epidemiology (WBE), and artificial intelligence.

This innovative approach leverages the capabilities of SERS to detect molecular markers at ultra-low concentrations in wastewater samples, providing unprecedented sensitivity and specificity in public health monitoring. By analyzing community wastewater, researchers can gather population-level health information without invasive individual testing, creating a comprehensive picture of community health status in real-time.
