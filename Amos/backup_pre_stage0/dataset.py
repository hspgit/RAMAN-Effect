import os

import numpy as np
import ramanspy as rp
import torch
from torch.utils.data import Dataset


class RamanDataset(Dataset):
    def __init__(self, X_path, y_path, label_mapping=None, apply_preprocessing=True):
        """
        Loads the Raman spectra data from numpy arrays.
        X shape: (N, L) -> expanded to (N, 1, L) for 1D CNNs
        y shape: (N,)
        """
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing data files: {X_path} or {y_path}")
            
        self.X = np.load(X_path).astype(np.float32)
        
        if apply_preprocessing:
            print(f"Applying RamanSPy preprocessing to {X_path}...")
            
            # Load wavenumbers for ramanspy
            wavenumbers_path = os.path.join(os.path.dirname(X_path), 'wavenumbers.npy')
            if os.path.exists(wavenumbers_path):
                wavenumbers = np.load(wavenumbers_path)
            else:
                wavenumbers = np.arange(self.X.shape[1])
                
            pipeline = rp.preprocessing.Pipeline([
                rp.preprocessing.despike.WhitakerHayes(),
                rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
                rp.preprocessing.baseline.ASPLS(),
                rp.preprocessing.normalise.MinMax()
            ])
            
            # ramanspy expects SpectralContainer or Spectrum objects
            raman_obj = rp.SpectralContainer(self.X, wavenumbers)
            
            # Apply the pipeline to the data
            processed_obj = pipeline.apply(raman_obj)
            self.X = processed_obj.spectral_data
            
            # Ensure it is a float32 numpy array
            if not isinstance(self.X, np.ndarray):
                self.X = np.array(self.X)
            self.X = self.X.astype(np.float32)

        # Add channel dimension for 1D CNN: (N, C, L) where C=1
        self.X = np.expand_dims(self.X, axis=1)
        
        raw_y = np.load(y_path)
        if label_mapping is None:
            unique_labels = sorted(np.unique(raw_y))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_mapping = label_mapping
            
        # Filter out samples with unseen labels
        valid_indices = [i for i, label in enumerate(raw_y) if label in self.label_mapping]
        if len(valid_indices) < len(raw_y):
            print(f"Warning: Dropped {len(raw_y) - len(valid_indices)} samples with unseen labels.")
            self.X = self.X[valid_indices]
            raw_y = raw_y[valid_indices]
            
        # Map labels to 0, 1, ..., C-1
        mapped_y = np.array([self.label_mapping[label] for label in raw_y])
        self.y = mapped_y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class AugmentedDataset(Dataset):
    """
    Wraps a PyTorch Dataset (or Subset) to apply on-the-fly augmentation.
    """
    def __init__(self, dataset, augment=False):
        self.dataset = dataset
        self.augment = augment
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        if self.augment:
            # Convert back to numpy for augmentation
            x_np = x.numpy()
            
            # 1. Random shift (roll)
            shift = np.random.randint(-5, 6)
            x_np = np.roll(x_np, shift, axis=-1)
            
            # 2. Random Gaussian noise
            noise = np.random.normal(0, 0.01, x_np.shape).astype(np.float32)
            x_np = x_np + noise
            
            # 3. Random scale
            scale = np.random.uniform(0.95, 1.05)
            x_np = x_np * scale
            
            x = torch.tensor(x_np)
            
        return x, y
