# Treatment Group Accuracy Results: Legacy CNN vs. ResNet

This document summarizes the **30-Class Isolate Accuracy** and **8-Class Empiric Antibiotic Treatment Group Accuracy** for our two primary convolutional architectures—[`LegacyCNN`](../legacy_baseline/model.py) and [`Raman1DCNN` (ResNet)](../src/model.py)—evaluated on the independent reference test set (`X_test.npy`, $N = 3,000$).

---

## 1. Experimental Setup

- **Pre-training Dataset**: `X_reference.npy` / `y_reference.npy` (60,000 spectra across 30 reference isolates, 15 epochs)
- **Fine-tuning Dataset**: `X_finetune.npy` / `y_finetune.npy` (3,000 spectra across 30 reference isolates, 5 epochs)
- **Test Dataset**: `X_test.npy` / `y_test.npy` (3,000 unseen spectra, 100 per isolate)
- **Preprocessing**: Raw spectra (`--no-preprocessing`), leveraging our finding in [`11_PREPROCESSING_IMPACT.md`](11_PREPROCESSING_IMPACT.md) that 1D CNNs extract richer high-frequency chemical features directly from raw Raman spectra
- **Evaluation Metric**: Post-hoc 30-to-8 mapping via `ATCC_GROUPINGS` in [`src/engine.py`](../src/engine.py) (see [`12_ANTIBIOTIC_TREATMENT_GROUPINGS.md`](12_ANTIBIOTIC_TREATMENT_GROUPINGS.md))

---

## 2. Overall Accuracy Comparison

| Model Architecture | Test Loss | 30-Class Isolate Accuracy | 8-Class Treatment Group Accuracy | Accuracy Gain (+%) |
| :--- | :---: | :---: | :---: | :---: |
| **[`LegacyCNN`](../legacy_baseline/model.py)** | **0.6493** | **82.23%** (2,467 / 3,000) | **96.53%** (2,896 / 3,000) | **+14.30%** |
| **[`Raman1DCNN` (ResNet)](../src/model.py)** | 0.6804 | **80.00%** (2,400 / 3,000) | **94.80%** (2,844 / 3,000) | **+14.80%** |
| *Paper Baseline (`bacteria-ID`)* | *—* | *82.2% – 82.5%* | *97.0% – 97.1%* | *+14.6%* |

---

## 3. Per-Treatment-Group Accuracy Breakdown

Evaluating accuracy within each of the 8 empiric antimicrobial groups reveals how well each architecture identifies the correct clinical treatment across different pathogen families:

| Group ID | Empiric Antibiotic | Target Pathogens | Test Spectra | Legacy CNN Accuracy | ResNet (`Raman1DCNN`) Accuracy |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **0** | **Meropenem** | Enteric Gram-negative rods (8 strains) | 800 | **96.75%** (774 / 800) | 96.00% (768 / 800) |
| **1** | **Ciprofloxacin** | *S. enterica* (1 strain) | 100 | **97.00%** (97 / 100) | 96.00% (96 / 100) |
| **2** | **TZP** | *P. aeruginosa* (2 strains) | 200 | **96.00%** (192 / 200) | 88.00% (176 / 200) |
| **3** | **Vancomycin** | *Staphylococci* — MSSA/MRSA/CoNS (7 strains) | 700 | 96.71% (677 / 700) | **97.29%** (681 / 700) |
| **4** | **Ceftriaxone** | *S. pneumoniae* (2 strains) | 200 | 93.00% (186 / 200) | **94.00%** (188 / 200) |
| **5** | **Penicillin** | *Streptococci* & *E. faecalis* (7 strains) | 700 | **95.86%** (671 / 700) | 91.29% (639 / 700) |
| **6** | **Daptomycin** | *E. faecium* (1 strain) | 100 | **100.00%** (100 / 100) | **100.00%** (100 / 100) |
| **7** | **Caspofungin** | *Candida* yeasts (2 strains) | 200 | **99.50%** (199 / 200) | 98.00% (196 / 200) |
| **All** | **Total / Overall** | **All 30 Reference Isolates** | **3,000** | **96.53%** (2,896 / 3,000) | **94.80%** (2,844 / 3,000) |

---

## 4. Key Takeaways

1. **Over 80% of "Errors" Are Within the Same Treatment Group**:
   - For **Legacy CNN**, out of 533 isolate-level misclassifications ($3,000 - 2,467$), **429 (80.5%)** occurred between strains belonging to the *exact same* antibiotic treatment group, leaving only 104 cross-group errors out of 3,000 spectra (**96.53% accuracy**).
   - For **ResNet**, out of 600 isolate-level misclassifications ($3,000 - 2,400$), **444 (74.0%)** occurred within the same treatment group (**94.80% accuracy**).
2. **Legacy CNN Matches the Published Benchmark**:
   - With just 15 pre-training epochs and 5 fine-tuning epochs on raw data, **Legacy CNN** achieves **82.23% isolate accuracy** and **96.53% treatment group accuracy**, virtually matching the `bacteria-ID` paper's reported 82.2% isolate / 97.0% treatment group benchmark.
3. **High-Confidence Pathogen Groups**:
   - Both models achieve **100.00%** accuracy on **Daptomycin** (*E. faecium*) and **98.00%–99.50%** on **Caspofungin** (*Candida* species), demonstrating near-perfect separation of fungal spectra and vancomycin-resistant enterococci from other bacterial classes.
   - **Legacy CNN** holds its strongest advantage over ResNet in **TZP** (*P. aeruginosa*: 96.00% vs. 88.00%) and **Penicillin** (*Streptococci* / *E. faecalis*: 95.86% vs. 91.29%).
