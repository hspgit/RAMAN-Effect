import numpy as np
import ramanspy as rp

X = np.load("data/X_test.npy").astype(np.float32)
wavenumbers = np.load("data/wavenumbers.npy")
current_obj = rp.Spectrum(X[0], wavenumbers)

steps = [
    ("1_Despiking", rp.preprocessing.despike.WhitakerHayes()),
    ("2_Denoising", rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3)),
]

results = []
for name, func in steps:
    current_obj = func.apply(current_obj)
    results.append(current_obj)

print("First result max:", results[0].spectral_data.max())
print("Second result max:", results[1].spectral_data.max())
print("Are they the same object?", results[0] is results[1])
print("Are the spectral_data the same?", np.array_equal(results[0].spectral_data, results[1].spectral_data))
