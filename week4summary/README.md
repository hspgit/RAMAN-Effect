# Raman Spectra Preprocessing Pipeline Visualizations

This folder contains individual visualizations of the effect of each preprocessing step on three example Raman spectra from the dataset (`X_test.npy`).
By overlaying the 'Before' and 'After' states, we can see exactly what each transformation does to the data.

## Raw Data

The original unprocessed Raman spectra.

![Raw Data Sample 0](images/0_Raw_Data_Sample_0.png)

![Raw Data Sample 50](images/0_Raw_Data_Sample_50.png)

![Raw Data Sample 100](images/0_Raw_Data_Sample_100.png)

## Despiking (Whitaker-Hayes)

Removes cosmic ray spikes from the spectrum.

![Despiking (Whitaker-Hayes) Sample 0](images/1_Despiking_Sample_0.png)

![Despiking (Whitaker-Hayes) Sample 50](images/1_Despiking_Sample_50.png)

![Despiking (Whitaker-Hayes) Sample 100](images/1_Despiking_Sample_100.png)

## Denoising (Savitzky-Golay)

Smoothes the signal to reduce high-frequency noise.

![Denoising (Savitzky-Golay) Sample 0](images/2_Denoising_Sample_0.png)

![Denoising (Savitzky-Golay) Sample 50](images/2_Denoising_Sample_50.png)

![Denoising (Savitzky-Golay) Sample 100](images/2_Denoising_Sample_100.png)

## Baseline Correction (ASPLS)

Removes the fluorescent baseline background. Notice how it behaves on this dataset!

![Baseline Correction (ASPLS) Sample 0](images/3_Baseline_Correction_Sample_0.png)

![Baseline Correction (ASPLS) Sample 50](images/3_Baseline_Correction_Sample_50.png)

![Baseline Correction (ASPLS) Sample 100](images/3_Baseline_Correction_Sample_100.png)

## Normalization (MinMax)

Scales the intensities to a fixed range (typically 0 to 1).

![Normalization (MinMax) Sample 0](images/4_Normalization_Sample_0.png)

![Normalization (MinMax) Sample 50](images/4_Normalization_Sample_50.png)

![Normalization (MinMax) Sample 100](images/4_Normalization_Sample_100.png)

