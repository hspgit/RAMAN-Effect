import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from legacy_baseline.model import LegacyCNN
from src.model import Raman1DCNN
from src.engine import ATCC_GROUPINGS, ANTIBIOTICS

def main():
    print("Evaluating models for Empiric Antibiotic Treatment Group Heatmaps...")
    
    # 1. Load trained models
    m_plain = LegacyCNN(num_classes=30)
    m_plain.load_state_dict(torch.load('models/legacy_cnn_finetuned.pth', map_location='cpu'))
    m_plain.eval()

    m_resnet = Raman1DCNN(num_classes=30)
    m_resnet.load_state_dict(torch.load('models/resnet_finetuned.pth', map_location='cpu'))
    m_resnet.eval()

    # 2. Load test set
    X = np.load('data/X_test.npy').astype(np.float32)
    y = np.load('data/y_test.npy').astype(np.int64)

    with torch.no_grad():
        x_t = torch.tensor(X).unsqueeze(1) if len(X.shape) == 2 else torch.tensor(X)
        p_plain_iso = m_plain(x_t).argmax(dim=1).numpy()
        p_resnet_iso = m_resnet(x_t).argmax(dim=1).numpy()

    # Map to 8 empiric groups
    y_grp = np.array([ATCC_GROUPINGS[int(i)] for i in y])
    p_plain_grp = np.array([ATCC_GROUPINGS[int(i)] for i in p_plain_iso])
    p_resnet_grp = np.array([ATCC_GROUPINGS[int(i)] for i in p_resnet_iso])

    group_labels = [
        "Meropenem\n(MEM)",
        "Ciprofloxacin\n(CIP)",
        "Pip.-tazo.\n(TZP)",
        "Vancomycin\n(VAN)",
        "Ceftriaxone\n(CRO)",
        "Penicillin\n(PEN)",
        "Daptomycin\n(DAP)",
        "Caspofungin\n(CAS)"
    ]

    short_labels = ["MEM", "CIP", "TZP", "VAN", "CRO", "PEN", "DAP", "CAS"]

    cm_plain = confusion_matrix(y_grp, p_plain_grp, labels=np.arange(8))
    cm_resnet = confusion_matrix(y_grp, p_resnet_grp, labels=np.arange(8))

    # Normalize by row (true labels)
    cm_plain_norm = cm_plain.astype('float') / cm_plain.sum(axis=1)[:, np.newaxis] * 100.0
    cm_resnet_norm = cm_resnet.astype('float') / cm_resnet.sum(axis=1)[:, np.newaxis] * 100.0

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cand = os.path.join(repo_root, 'paper', 'overleaf', 'figures')
    output_dir = cand if os.path.exists(cand) else os.path.join(repo_root, 'overleaf', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # IEEE styling settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
    })

    # Function to draw annotated heatmap
    def plot_single_cm(cm_raw, cm_norm, title, acc_total, filename):
        fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=8)

        ax.set(xticks=np.arange(8),
               yticks=np.arange(8),
               xticklabels=short_labels,
               yticklabels=short_labels)
        
        ax.set_xlabel('Predicted Treatment Group', fontweight='bold', fontsize=8.5)
        ax.set_ylabel('Ground Truth Treatment Group', fontweight='bold', fontsize=8.5)
        ax.set_title(f'{title} (Overall: {acc_total:.2f}%)', fontsize=9, fontweight='bold', pad=8)

        # Loop over data dimensions and create text annotations.
        thresh = 50.0
        for i in range(8):
            for j in range(8):
                val = cm_norm[i, j]
                count = cm_raw[i, j]
                if val >= 0.5:
                    text_str = f"{val:.1f}%\n({count})" if val >= 10.0 else f"{val:.1f}%\n({count})"
                else:
                    text_str = f"{count}" if count > 0 else "0"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text_str,
                        ha="center", va="center",
                        color=color, fontsize=5.8, fontweight='semibold' if i == j else 'normal')

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}.pdf and .png")

    acc_plain = (p_plain_grp == y_grp).mean() * 100.0
    acc_resnet = (p_resnet_grp == y_grp).mean() * 100.0

    plot_single_cm(cm_plain, cm_plain_norm, 'Plain 1D CNN', acc_plain, 'heatmap_treatment_plain')
    plot_single_cm(cm_resnet, cm_resnet_norm, '1D ResNet', acc_resnet, 'heatmap_treatment_resnet')

    # Combined Side-by-Side Plot (Figure*)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=300)
    
    for ax, cm_raw, cm_norm, title, acc in [
        (ax1, cm_plain, cm_plain_norm, '(a) Plain 1D CNN', acc_plain),
        (ax2, cm_resnet, cm_resnet_norm, '(b) 1D ResNet', acc_resnet)
    ]:
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
        ax.set(xticks=np.arange(8),
               yticks=np.arange(8),
               xticklabels=short_labels,
               yticklabels=short_labels)
        ax.set_xlabel('Predicted Treatment', fontweight='bold', fontsize=8)
        ax.set_ylabel('True Treatment', fontweight='bold', fontsize=8)
        ax.set_title(f'{title} (Accuracy: {acc:.2f}%)', fontsize=8.5, fontweight='bold', pad=6)

        thresh = 50.0
        for i in range(8):
            for j in range(8):
                val = cm_norm[i, j]
                count = cm_raw[i, j]
                if val >= 0.5:
                    text_str = f"{val:.1f}%\n({count})"
                else:
                    text_str = f"{count}" if count > 0 else "-"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text_str,
                        ha="center", va="center",
                        color=color, fontsize=5.0, fontweight='semibold' if i == j else 'normal')

    fig.subplots_adjust(right=0.88, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Class Accuracy (%)', rotation=-90, va="bottom", fontsize=8)

    plt.savefig(os.path.join(output_dir, 'heatmap_treatment_combined.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'heatmap_treatment_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved heatmap_treatment_combined.pdf and .png")

if __name__ == '__main__':
    main()
