# Empiric Antibiotic Treatment Groupings: What and Why

In the foundational `bacteria-ID` study (*"Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning"*, Ho et al., Nature Communications 2019), the authors evaluate model performance using two complementary accuracy metrics:
1. **30-Class Isolate Accuracy** (identifying the exact bacterial or yeast strain out of 30 reference isolates).
2. **8-Class Empiric Antibiotic Treatment Group Accuracy** (identifying the correct first-line antibiotic treatment group).

---

## 1. Why Group Isolates by Antibiotic Treatment?

### Clinical Motivation: Empiric Therapy
When a patient presents with a severe bacterial or fungal infection (such as sepsis or bloodstream infection), clinicians must administer **empiric antibiotic therapy** immediately—long before traditional culture and susceptibility testing (which can take 24–72 hours) completes.

From a clinical decision-making standpoint, a physician does **not** need to distinguish between every individual sub-strain (e.g., *E. coli 1* vs. *E. coli 2*, or *MSSA 1* vs. *MSSA 3*) if both strains are treated with the exact same first-line empiric antimicrobial agent.

### Spectral Similarity Within Genera
Biochemically, closely related strains within the same species or genus share nearly identical cell-wall compositions and vibrational modes, producing highly overlapping Raman spectra. Consequently:
- A deep learning classifier may occasionally confuse *MSSA 1* with *MSSA 2*, or *E. coli 1* with *K. pneumoniae 1*.
- While this counts as an **error** under strict 30-class isolate accuracy, it is **clinically inconsequential** if both organisms belong to the same empiric antibiotic treatment group.

Grouping the 30 isolates into their 8 standard-of-care empiric antibiotic classes measures the **true clinical utility** of the Raman + Deep Learning diagnostic pipeline.

---

## 2. The 30-to-8 Mapping (`ATCC_GROUPINGS`)

In [`bacteria-ID/config.py`](../bacteria-ID/config.py) (and integrated into [`src/engine.py`](../src/engine.py)), the 30 reference isolates (`0..29`) are mapped into **8 empiric antimicrobial treatment groups** (`0..7`):

| Group ID | Empiric Antimicrobial | Target Pathogen Category | Mapped Reference Isolates (`STRAINS` IDs) | Test Set Samples |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **Meropenem** | Gram-negative enteric rods (*Enterobacterales*) | *E. coli 1* (`3`), *E. coli 2* (`4`), *K. pneumoniae 1* (`9`), *K. pneumoniae 2* (`10`), *K. aerogenes* (`2`), *E. cloacae* (`8`), *P. mirabilis* (`11`), *S. marcescens* (`22`) | 800 |
| **1** | **Ciprofloxacin** | *Salmonella* species | *S. enterica* (`19`) | 100 |
| **2** | **TZP** (Piperacillin-tazobactam) | *Pseudomonas aeruginosa* | *P. aeruginosa 1* (`12`), *P. aeruginosa 2* (`13`) | 200 |
| **3** | **Vancomycin** | *Staphylococcus* species (MSSA, MRSA, CoNS) | *MSSA 1* (`14`), *MSSA 2* (`18`), *MSSA 3* (`15`), *MRSA 1 (isogenic)* (`16`), *MRSA 2* (`17`), *S. epidermidis* (`20`), *S. lugdunensis* (`21`) | 700 |
| **4** | **Ceftriaxone** | *Streptococcus pneumoniae* | *S. pneumoniae 1* (`24`), *S. pneumoniae 2* (`23`) | 200 |
| **5** | **Penicillin** | Non-pneumococcal *Streptococci* & *E. faecalis* | *Group A Strep.* (`26`), *Group B Strep.* (`27`), *Group C Strep.* (`28`), *Group G Strep.* (`29`), *S. sanguinis* (`25`), *E. faecalis 1* (`6`), *E. faecalis 2* (`7`) | 700 |
| **6** | **Daptomycin** | *Enterococcus faecium* (often vancomycin-resistant) | *E. faecium* (`5`) | 100 |
| **7** | **Caspofungin** | *Candida* fungal species (yeasts) | *C. albicans* (`0`), *C. glabrata* (`1`) | 200 |

---

## 3. How Treatment Group Accuracy is Evaluated

The `bacteria-ID` repository uses two evaluation protocols depending on the dataset:

### A. Reference Dataset: Post-Hoc 30-to-8 Mapping (`2_reference_prediction.ipynb`)
1. The neural network is trained and fine-tuned as a **30-class isolate classifier** (`num_classes = 30`).
2. For each spectrum $i$ in `X_test.npy` ($N = 3,000$, 100 spectra per isolate), the model predicts an isolate label $\hat{y}_i \in \{0, \dots, 29\}$.
3. Both the ground-truth isolate label $y_i$ and predicted isolate label $\hat{y}_i$ are mapped post-hoc to their corresponding antibiotic treatment groups using `ATCC_GROUPINGS`:
   $$\text{Group}(y_i) \in \{0, \dots, 7\}, \quad \text{Group}(\hat{y}_i) \in \{0, \dots, 7\}$$
4. **Treatment Group Accuracy** is the percentage of test spectra where $\text{Group}(\hat{y}_i) = \text{Group}(y_i)$.

### B. Clinical Patient Datasets: Direct 8-Class Head + 10-Spectra Majority Vote (`3_clinical_finetuning.ipynb`)
For the clinical patient isolates (`X_2018clinical.npy` and `X_2019clinical.npy`, 25 patients across 5 treatment groups):
1. **8-Class Output Layer**: The model is initialized with `n_classes = 8`, directly predicting the treatment group.
2. **Patient-Level Split**: For each of the 5 treatment groups (5 patients per group), 3 patients are assigned to training, 1 to validation, and 1 to testing, sampling only 10 spectra per patient during fine-tuning.
3. **10-Spectra Mode (Bootstrapped Majority Voting)**: To simulate a culture-free clinical workflow where only 10 bacterial cells are measured from a patient sample, 10,000 Monte Carlo trials are run per test patient. In each trial, 10 spectrum predictions are randomly drawn from that patient's spectra pool and combined via majority vote (`scipy.stats.mode`) to produce the patient-level treatment recommendation.
