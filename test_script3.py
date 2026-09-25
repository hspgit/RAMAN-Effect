import numpy as np
X = np.load("data/X_test.npy")
print("Min of X_test:", X.min())
print("Max of X_test:", X.max())
print("Min of first 10 samples:", X[:10].min(axis=1))
print("Max of first 10 samples:", X[:10].max(axis=1))
