# Raman Spectra Preprocessing

Raw Raman spectroscopy data is often affected by various sources of noise, artifacts, and background signals (such as fluorescence and cosmic rays). To ensure that our machine learning models learn the true underlying chemical signatures rather than instrument artifacts, we apply a comprehensive preprocessing pipeline using the `ramanspy` library.

The preprocessing pipeline is implemented in `src/dataset.py` and consists of the following sequential steps:

## 1. Despiking (Whitaker-Hayes)
**Method:** `ramanspy.preprocessing.despike.WhitakerHayes()`

- **What it does:** Identifies and removes sharp, isolated spikes in the spectral data.
- **Why it is needed:** When taking Raman measurements, high-energy particles (like cosmic rays) hitting the detector can cause massive, artificial spikes in intensity at random wavenumbers. These spikes are not representative of the sample's chemistry and can severely confuse the model. The Whitaker-Hayes algorithm detects these unnatural anomalies and interpolates the data to remove them smoothly without destroying the real Raman peaks.

## 2. Denoising (Savitzky-Golay Filter)
**Method:** `ramanspy.preprocessing.denoise.SavGol(window_length=9, polyorder=3)`

- **What it does:** Smooths the signal by fitting a low-degree polynomial (order 3) to a sliding window (length 9) of data points.
- **Why it is needed:** Instruments often introduce high-frequency, random, white noise (sensor noise or thermal noise). Simply averaging the data would flatten the important sharp Raman peaks. The Savitzky-Golay filter effectively smooths out the small jitters (noise) while preserving the height, width, and area of the actual chemical peaks, increasing the signal-to-noise ratio.

## 3. Baseline Correction (ASPLS)
**Method:** `ramanspy.preprocessing.baseline.ASPLS()`

- **What it does:** Estimates and subtracts the slowly varying background signal (baseline) using Asymmetric Least Squares Smoothing.
- **Why it is needed:** Biological samples (like bacteria) frequently exhibit auto-fluorescence when illuminated by a laser. This fluorescence creates a massive, broad hump across the entire spectrum, often dwarfing the actual Raman peaks that sit on top of it. ASPLS accurately maps out this underlying curved baseline so it can be subtracted out, leaving behind a "flat" spectrum where all the peaks start near zero.

## 4. Normalization (Min-Max)
**Method:** `ramanspy.preprocessing.normalise.MinMax()`

- **What it does:** Scales the entire spectrum so that the minimum intensity becomes 0 and the maximum intensity becomes 1.
- **Why it is needed:** Different samples might have different overall intensities due to variations in sample concentration, laser focus, or exposure time. Min-Max normalization ensures that all spectra are evaluated on the exact same scale. This uniformity is crucial for deep learning models (like our 1D CNN), as it helps the network converge faster and focus on the *relative* heights and shapes of the peaks rather than their absolute magnitudes.
