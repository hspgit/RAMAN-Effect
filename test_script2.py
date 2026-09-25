import numpy as np

X = np.load("data/X_test.npy").astype(np.float32)
print("Raw max:", X[0].max())
print("Raw min:", X[0].min())
