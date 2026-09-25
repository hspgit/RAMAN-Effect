import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt

X = np.load("data/X_test.npy").astype(np.float32)
wavenumbers = np.load("data/wavenumbers.npy")

steps = [
    ("1_Despiking", "Despiking", rp.preprocessing.despike.WhitakerHayes()),
    ("2_Denoising", "Denoising", rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3)),
    ("3_Baseline_Correction", "Baseline Correction", rp.preprocessing.baseline.ASPLS()),
    ("4_Normalization", "Normalization", rp.preprocessing.normalise.MinMax())
]

current_obj = rp.Spectrum(X[0], wavenumbers)
prev_data = current_obj.spectral_data.copy()

current_obj = steps[2][2].apply(current_obj)
curr_data = current_obj.spectral_data

baseline = prev_data - curr_data
print("Baseline max:", baseline.max(), "min:", baseline.min())
