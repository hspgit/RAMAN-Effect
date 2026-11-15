import numpy as np
import ramanspy as rp

X = np.load("data/X_test.npy").astype(np.float32)
wavenumbers = np.load("data/wavenumbers.npy")

steps = [
    ("1_Despiking", rp.preprocessing.despike.WhitakerHayes()),
    ("2_Denoising", rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3)),
    ("3_Baseline_Correction", rp.preprocessing.baseline.ASPLS()),
    ("4_Normalization", rp.preprocessing.normalise.MinMax())
]

for idx in [0, 10, 100]:
    print(f"\n--- Sample {idx} ---")
    current_obj = rp.Spectrum(X[idx], wavenumbers)
    print(f"Raw: min={current_obj.spectral_data.min():.3f}, max={current_obj.spectral_data.max():.3f}")
    
    for name, func in steps:
        prev_data = current_obj.spectral_data.copy()
        current_obj = func.apply(current_obj)
        curr_data = current_obj.spectral_data
        
        diff = np.abs(curr_data - prev_data).mean()
        print(f"{name}: diff_from_prev={diff:.3f}, min={curr_data.min():.3f}, max={curr_data.max():.3f}")

