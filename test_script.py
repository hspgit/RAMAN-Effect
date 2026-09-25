import numpy as np
import ramanspy as rp

X = np.load("data/X_test.npy").astype(np.float32)
wavenumbers = np.load("data/wavenumbers.npy")
current_obj = rp.Spectrum(X[0], wavenumbers)

steps = [
    ("1_Despiking", rp.preprocessing.despike.WhitakerHayes()),
    ("2_Denoising", rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3)),
    ("3_Baseline_Correction", rp.preprocessing.baseline.ASPLS()),
    ("4_Normalization", rp.preprocessing.normalise.MinMax())
]

for name, func in steps:
    current_obj = func.apply(current_obj)
    print(f"{name}: shape={current_obj.spectral_data.shape}, min={current_obj.spectral_data.min()}, max={current_obj.spectral_data.max()}")

