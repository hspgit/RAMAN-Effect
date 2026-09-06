import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import ramanspy as rp

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

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Raman1DCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(Raman1DCNN, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_epoch(model, device, train_loader, optimizer, criterion, epoch, phase="Train"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'{phase} Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\t'
                  f'Acc: {100. * correct / total:.2f}%')

def evaluate(model, device, data_loader, criterion, phase="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    acc = 100. * correct / len(data_loader.dataset)
    print(f'\n{phase} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} '
          f'({acc:.2f}%)\n')
    return acc

def main():
    parser = argparse.ArgumentParser(description='Deep Learning for Raman Spectra Classification')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs for reference training (default: 15)')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='number of epochs for finetuning (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='directory containing the .npy files')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='directory to save model checkpoints')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # 1. Load Reference Data
    print("Loading reference dataset...")
    ref_dataset = RamanDataset(os.path.join(args.data_dir, 'X_reference.npy'),
                               os.path.join(args.data_dir, 'y_reference.npy'))
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=True)
    
    num_classes = len(np.unique(ref_dataset.y))
    print(f"Detected {num_classes} classes.")

    # 2. Initialize Model
    model = Raman1DCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Train on Reference Data
    print("\n--- Starting Reference Training ---")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, ref_loader, optimizer, criterion, epoch, phase="Pre-train")
        scheduler.step()
    
    ref_model_path = os.path.join(args.save_dir, 'raman_reference_model.pth')
    torch.save(model.state_dict(), ref_model_path)
    print(f"Reference model saved to {ref_model_path}")

    # 4. Finetune Data (if available)
    finetune_x = os.path.join(args.data_dir, 'X_finetune.npy')
    finetune_y = os.path.join(args.data_dir, 'y_finetune.npy')
    
    if os.path.exists(finetune_x) and os.path.exists(finetune_y) and args.finetune_epochs > 0:
        print("\n--- Starting Finetuning ---")
        finetune_dataset = RamanDataset(finetune_x, finetune_y, label_mapping=ref_dataset.label_mapping)
        finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Optionally reduce learning rate for finetuning
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)
        
        for epoch in range(1, args.finetune_epochs + 1):
            train_epoch(model, device, finetune_loader, optimizer_ft, criterion, epoch, phase="Finetune")
            scheduler_ft.step()
            
        ft_model_path = os.path.join(args.save_dir, 'raman_finetuned_model.pth')
        torch.save(model.state_dict(), ft_model_path)
        print(f"Finetuned model saved to {ft_model_path}")
    else:
        print("\nSkipping finetuning (files not found or finetune-epochs=0).")

    # 5. Evaluate on Test Data
    test_x = os.path.join(args.data_dir, 'X_test.npy')
    test_y = os.path.join(args.data_dir, 'y_test.npy')
    
    if os.path.exists(test_x) and os.path.exists(test_y):
        print("\n--- Evaluating on Test Set ---")
        test_dataset = RamanDataset(test_x, test_y, label_mapping=ref_dataset.label_mapping)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
        evaluate(model, device, test_loader, criterion, phase="Test")
    else:
        print("\nTest data not found, skipping evaluation.")

if __name__ == '__main__':
    main()
