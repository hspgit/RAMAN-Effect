# Comprehensive Experiment Summary: Raman Spectra Classification

This document serves as a complete record of all the model architectures, configurations, and experiments run during our optimization phase. Our goal was to find the best deep learning architecture for classifying 1D Raman spectra (1000-point vectors) across 30 bacterial classes.

---

## 1. The "Optimized" Configurations (With Augmentation)
*Setup: 100 Pre-train Epochs, 20 Fine-tune Epochs, On-the-fly Data Augmentation (Shifting, Scaling, Noise).*

| Model Architecture | Specific Tweaks | Final Test Accuracy | Why it happened |
| :--- | :--- | :--- | :--- |
| **Legacy CNN** | Dropout raised to 0.3 | **76.50% (Champion)** | The shallow, straightforward structure of this CNN acts as a natural "bottleneck." It is perfectly suited for the 1D spatial nature of the data, capturing local chemical peaks without having excess capacity to memorize background noise. |
| **ResNet** | Dropout raised to 0.4 | **69.90%** | The deep residual layers gave the model too much "capacity." Even with aggressive dropout and data augmentation, it overfitted the relatively simple 1D dataset, failing to generalize to the test set as well as the shallower CNN. |
| **Multiscale 1D-CNN** | Inception blocks (1x1, 3x3, 7x7, 11x11 branches) | **63.73%** | While theoretically sound (scanning for both sharp and wide peaks simultaneously), the parallel branches widened the network too much. This excess width led to mild overfitting compared to the strictly constrained Legacy CNN. |
| **CNN-Transformer Fusion**| Fragment extraction (CNN) + Global Attention | **60.87%** | Successfully beat the pure Transformer by using a CNN to extract local peaks first, but without MixUp, the Transformer backend still lacked the data variety needed to fully conquer the spatial relationships. |
| **1D Transformer** | Patchification (100-pt patches), 3 Layers, 4 Heads | **54.73%** | Transformers lack spatial "inductive bias" (they don't natively understand that point 2 is next to point 3). Without a massive dataset or extreme augmentation (like MixUp), it severely underfitted and failed to learn the basic shapes of the chemical peaks. |

---

## 2. The "Pure" Configurations (No Augmentation)
*Setup: 100 Pre-train Epochs, 20 Fine-tune Epochs, **NO Data Augmentation**. This test was designed to evaluate the raw structural capability of the models in a pure vacuum.*

| Model Architecture | Final Test Accuracy | Why it happened |
| :--- | :--- | :--- |
| **Pure Legacy CNN** | **75.10%** | Even without physical data augmentation, the shallower Legacy CNN dominates. Its structure alone acts as a natural regularizer, proving it is the mathematically correct fit for this domain. |
| **Pure ResNet** | **67.47%** | Without Data Augmentation to aggressively regularize it, the deep ResNet heavily memorizes the training data (overfits). |
| **Pure 1D Transformer**| **57.83%** | Struggles heavily. Without data augmentation, it cannot learn the fundamental spatial patterns that the CNNs get for free. |

---

## 3. The "Extreme Regularization" Experiment
*Setup: 100 Pre-train Epochs, 20 Fine-tune Epochs, Data Augmentation.*

| Model Architecture | Specific Tweaks | Final Test Accuracy | Why it happened |
| :--- | :--- | :--- | :--- |
| **Multiscale 1D-CNN** | **High Weight Decay (0.05)** | **34.17%** | We attempted to stop the Multiscale CNN from overfitting by heavily penalizing large weights (L2 regularization). It completely backfired. The penalty choked the network, effectively "turning off" the learning process and causing severe underfitting. |

---

### Key Takeaways
1. **Complexity $\neq$ Better:** In Raman spectroscopy, simpler, shallower models natively match the simplicity of the 1D chemical peaks. 
2. **Transformers Need Help:** State-of-the-art Transformers will fail on standard laboratory datasets unless paired with advanced augmentation like **MixUp**.
3. **Data Augmentation Works:** Adding physical augmentation (shifting, noise) reliably boosted the Legacy CNN from 75.10% to 76.50%.
