# The Impact of Preprocessing on Raman Spectra Classification

During our optimization phase, we applied a standard Raman spectroscopy preprocessing pipeline using the `ramanspy` library before feeding the 1D spectra into our deep learning models. 

This standard pipeline included:
1. **Despiking** (Whitaker-Hayes)
2. **Denoising / Smoothing** (Savitzky-Golay filter: window length=9, polyorder=3)
3. **Baseline Correction** (ASPLS)
4. **Normalization** (MinMax)

We hypothesized that providing "cleaner" data would improve the neural network's ability to classify the 30 bacterial classes. However, an experiment isolating the impact of this pipeline yielded surprising results.

## Experiment Results
*Setup: Legacy CNN, 100 Pre-train Epochs, 20 Fine-tune Epochs, On-the-fly Data Augmentation (Shifting, Scaling, Noise).*

| Pipeline Setup | Final Test Accuracy |
| :--- | :--- |
| **With Preprocessing** | 76.50% |
| **Without Preprocessing (Raw Data)** | **85.30%** |

## Conclusion
By **removing** the preprocessing pipeline and passing the raw spectra (plus augmentation) directly into the model, the Legacy CNN's test accuracy leaped by nearly **9%**.

### Why did this happen?
While traditional machine learning (like PCA + SVM) often strictly requires smoothed, baseline-corrected spectra, Deep Learning architectures like CNNs are designed to act as their own feature extractors and filters. 

1. **Loss of Signal:** The Savitzky-Golay filter and baseline correction likely smoothed out subtle but critical high-frequency chemical peaks that the CNN was using to differentiate between closely related bacterial strains.
2. **Feature Extraction:** A Convolutional Neural Network natively learns to suppress noise and ignore shifting baselines if given enough varied data. Hard-coding these corrections beforehand restricted the information available to the network.

**Takeaway:** Deep learning models excel on raw data. In future experiments, we will rely on raw spectra and data augmentation rather than classical preprocessing pipelines.
