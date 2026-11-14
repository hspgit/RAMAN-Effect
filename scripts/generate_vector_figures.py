import os
import sys
import re

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
# Strain dictionary from bacteria-ID/config.py
STRAINS = {
    0: "C. albicans", 1: "C. glabrata", 2: "K. aerogenes", 3: "E. coli 1",
    4: "E. coli 2", 5: "E. faecium", 6: "E. faecalis 1", 7: "E. faecalis 2",
    8: "E. cloacae", 9: "K. pneumoniae 1", 10: "K. pneumoniae 2", 11: "P. mirabilis",
    12: "P. aeruginosa 1", 13: "P. aeruginosa 2", 14: "MSSA 1", 15: "MSSA 3",
    16: "MRSA 1 (isog.)", 17: "MRSA 2", 18: "MSSA 2", 19: "S. enterica",
    20: "S. epidermidis", 21: "S. lugdunensis", 22: "S. marcescens", 23: "S. pneumoniae 2",
    24: "S. pneumoniae 1", 25: "S. sanguinis", 26: "Group A Strep.", 27: "Group B Strep.",
    28: "Group C Strep.", 29: "Group G Strep."
}

def generate_confusion_matrices():
    print("Generating 30-class confusion matrix vector PDFs...")
    output_dir = 'overleaf/figures'
    os.makedirs(output_dir, exist_ok=True)

    X_test = np.load('data/X_test.npy').astype(np.float32)
    y_test = np.load('data/y_test.npy').astype(np.int64)
    x_t = torch.tensor(X_test).unsqueeze(1)

    m_plain = LegacyCNN(num_classes=30)
    m_plain.load_state_dict(torch.load('models/legacy_cnn_finetuned.pth', map_location='cpu'))
    m_plain.eval()

    m_resnet = Raman1DCNN(num_classes=30)
    m_resnet.load_state_dict(torch.load('models/resnet_finetuned.pth', map_location='cpu'))
    m_resnet.eval()

    with torch.no_grad():
        p_plain = m_plain(x_t).argmax(dim=1).numpy()
        p_resnet = m_resnet(x_t).argmax(dim=1).numpy()

    acc_plain = (p_plain == y_test).mean() * 100.0
    acc_resnet = (p_resnet == y_test).mean() * 100.0

    cm_plain = confusion_matrix(y_test, p_plain, labels=np.arange(30))
    cm_resnet = confusion_matrix(y_test, p_resnet, labels=np.arange(30))

    class_names = [STRAINS[i] for i in range(30)]

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 6.5,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 5.2,
        'ytick.labelsize': 5.2,
    })

    for cm, title, acc, filename in [
        (cm_plain, 'Plain 1D CNN', acc_plain, 'confusion_matrix_legacy_cnn'),
        (cm_resnet, '1D ResNet', acc_resnet, 'confusion_matrix_resnet')
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=300)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.set_ylabel('Test Count (per 100)', rotation=-90, va="bottom", fontsize=7.5)

        ax.set(xticks=np.arange(30),
               yticks=np.arange(30),
               xticklabels=class_names,
               yticklabels=class_names)
        
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        ax.set_xlabel('Predicted Isolate', fontweight='bold', fontsize=8)
        ax.set_ylabel('True Isolate', fontweight='bold', fontsize=8)
        ax.set_title(f'{title} Confusion Matrix (Acc: {acc:.2f}%)', fontweight='bold', fontsize=8.5, pad=6)

        fig.tight_layout()
        pdf_path = os.path.join(output_dir, f'{filename}.pdf')
        png_path = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {pdf_path}")

def generate_loss_curves():
    print("Generating loss curves vector PDF...")
    output_dir = 'overleaf/figures'

    def parse_log(path):
        epochs = []
        losses = []
        with open(path) as f:
            for line in f:
                m = re.search(r'Pre-train Epoch:\s*(\d+)\s*\[38400/40200.*?Loss:\s*([\d\.]+)', line)
                if m:
                    epochs.append(int(m.group(1)))
                    losses.append(float(m.group(2)))
        return epochs, losses

    ep_c, loss_c = parse_log('results/cnn/legacy_cnn_100_epochs_augmented.log')
    ep_r, loss_r = parse_log('results/resnet/resnet_100_epochs_augmented.log')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'lines.linewidth': 1.6,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)
    ax.plot(ep_c, loss_c, color='#1f77b4', label='Plain 1D CNN')
    ax.plot(ep_r, loss_r, color='#d62728', linestyle='--', label='1D ResNet')

    ax.set_xlabel('Pre-training Epoch')
    ax.set_ylabel('Cross-Entropy Training Loss')
    ax.set_title('Training Loss Trajectory', fontweight='bold')
    ax.set_xlim([1, 100])
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True)
    fig.tight_layout()

    pdf_path = os.path.join(output_dir, 'loss_curves.pdf')
    png_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {pdf_path}")

if __name__ == '__main__':
    generate_confusion_matrices()
    generate_loss_curves()
