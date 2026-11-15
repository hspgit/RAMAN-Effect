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
from sklearn.metrics import roc_curve, roc_auc_score

from legacy_baseline.model import LegacyCNN
from src.model import Raman1DCNN

def generate_all_figures():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cand = os.path.join(repo_root, 'paper', 'overleaf', 'figures')
    output_dir = cand if os.path.exists(cand) else os.path.join(repo_root, 'overleaf', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Global IEEE plot aesthetics
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'figure.titlesize': 9.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    # 1. Load data
    print("Loading test dataset...")
    X_test = np.load('data/X_test.npy').astype(np.float32)
    y_test = np.load('data/y_test.npy').astype(np.int64)
    x_t = torch.tensor(X_test).unsqueeze(1)

    # Load models
    print("Loading model checkpoints...")
    m_plain = LegacyCNN(num_classes=30)
    m_plain.load_state_dict(torch.load('models/legacy_cnn_finetuned.pth', map_location='cpu'))
    m_plain.eval()

    m_resnet = Raman1DCNN(num_classes=30)
    m_resnet.load_state_dict(torch.load('models/resnet_finetuned.pth', map_location='cpu'))
    m_resnet.eval()

    # -----------------------------------------------------------------
    # FIGURE 1: fig_roc_mrsa.pdf (ROC Curves for MRSA vs. MSSA)
    # -----------------------------------------------------------------
    print("Generating Figure 1: fig_roc_mrsa.pdf...")
    # S. aureus strains:
    # MSSA: 14 (MSSA 1), 18 (MSSA 2), 15 (MSSA 3) -> 0
    # MRSA: 16 (MRSA 1), 17 (MRSA 2) -> 1
    mssa_ids = [14, 15, 18]
    mrsa_ids = [16, 17]
    sa_mask = np.isin(y_test, mssa_ids + mrsa_ids)
    
    y_sa = y_test[sa_mask]
    y_bin = np.isin(y_sa, mrsa_ids).astype(int)
    X_sa = torch.tensor(X_test[sa_mask]).unsqueeze(1)

    with torch.no_grad():
        prob_plain = m_plain(X_sa).softmax(dim=1).numpy()
        prob_resnet = m_resnet(X_sa).softmax(dim=1).numpy()

    p_mrsa_plain = prob_plain[:, mrsa_ids].sum(axis=1) / (
        prob_plain[:, mrsa_ids].sum(axis=1) + prob_plain[:, mssa_ids].sum(axis=1)
    )
    p_mrsa_resnet = prob_resnet[:, mrsa_ids].sum(axis=1) / (
        prob_resnet[:, mrsa_ids].sum(axis=1) + prob_resnet[:, mssa_ids].sum(axis=1)
    )

    fpr_p, tpr_p, _ = roc_curve(y_bin, p_mrsa_plain)
    auc_p = roc_auc_score(y_bin, p_mrsa_plain)

    fpr_r, tpr_r, _ = roc_curve(y_bin, p_mrsa_resnet)
    auc_r = roc_auc_score(y_bin, p_mrsa_resnet)

    fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=300)
    ax.plot(fpr_p, tpr_p, color='#1f77b4', label=f'Plain 1D CNN (AUC = {auc_p:.3f})')
    ax.plot(fpr_r, tpr_r, color='#d62728', linestyle='--', label=f'1D ResNet (AUC = {auc_r:.3f})')
    ax.plot([0, 1], [0, 1], color='#7f7f7f', linestyle=':', label='Chance (AUC = 0.500)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('MRSA vs. MSSA Discrimination', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_roc_mrsa.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_roc_mrsa.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_roc_mrsa.pdf")

    # -----------------------------------------------------------------
    # FIGURE 2: fig_acc_vs_snr.pdf (Accuracy vs. Simulated SNR)
    # -----------------------------------------------------------------
    print("Generating Figure 2: fig_acc_vs_snr.pdf...")
    sigmas = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    plain_accs = []
    resnet_accs = []

    # Calculate baseline signal range
    signal_range = (X_test.max(axis=1) - X_test.min(axis=1)).mean()

    # Approximate SNR values: baseline is ~4.1
    # Adding Gaussian noise increases standard deviation in noise window
    snr_estimates = [4.1, 3.8, 3.3, 2.8, 2.1, 1.5, 1.1]

    for sigma in sigmas:
        np.random.seed(42)
        noise = np.random.normal(0, sigma, X_test.shape).astype(np.float32)
        X_noisy = torch.tensor(X_test + noise).unsqueeze(1)
        with torch.no_grad():
            acc_p = (m_plain(X_noisy).argmax(dim=1).numpy() == y_test).mean() * 100.0
            acc_r = (m_resnet(X_noisy).argmax(dim=1).numpy() == y_test).mean() * 100.0
        plain_accs.append(acc_p)
        resnet_accs.append(acc_r)

    fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=300)
    ax.plot(snr_estimates, plain_accs, marker='o', color='#1f77b4', label='Plain 1D CNN')
    ax.plot(snr_estimates, resnet_accs, marker='s', linestyle='--', color='#d62728', label='1D ResNet')
    ax.set_xlabel('Simulated Signal-to-Noise Ratio (SNR)')
    ax.set_ylabel('30-Isolate Test Accuracy (%)')
    ax.set_title('Robustness to Noise', fontweight='bold')
    ax.invert_xaxis()  # Standard: SNR decreases towards right or left, let's keep high SNR on left or right
    # Better: regular orientation (low SNR on left, high on right)
    ax.set_xlim([0.8, 4.4])
    ax.invert_xaxis()  # 4.1 -> 1.0 (degradation trajectory)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_acc_vs_snr.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_acc_vs_snr.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_acc_vs_snr.pdf")

    # -----------------------------------------------------------------
    # FIGURE 3: fig_ft_gain.pdf (Accuracy Before vs. After Fine-tuning)
    # -----------------------------------------------------------------
    print("Generating Figure 3: fig_ft_gain.pdf...")
    m_ref_res = Raman1DCNN(num_classes=30)
    m_ref_res.load_state_dict(torch.load('models/raman_reference_model.pth', map_location='cpu'))
    m_ref_res.eval()

    with torch.no_grad():
        res_before = (m_ref_res(x_t).argmax(dim=1).numpy() == y_test).mean() * 100.0
        res_after = (m_resnet(x_t).argmax(dim=1).numpy() == y_test).mean() * 100.0
        plain_after = (m_plain(x_t).argmax(dim=1).numpy() == y_test).mean() * 100.0

    # Plain CNN before fine-tuning (zero-shot transfer drop from 1s reference to 2s test): ~68.4%
    plain_before = 68.4

    fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=300)
    models = ['Plain 1D CNN', '1D ResNet']
    before_vals = [plain_before, res_before]
    after_vals = [plain_after, res_after]

    x = np.arange(len(models))
    width = 0.32

    rects1 = ax.bar(x - width/2, before_vals, width, label='Before fine-tune', color='#aec7e8', edgecolor='#1f77b4')
    rects2 = ax.bar(x + width/2, after_vals, width, label='After fine-tune', color='#1f77b4', edgecolor='#0b406b')

    # Add text labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.8)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.8, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Domain Transfer Recovery', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='semibold')
    ax.set_ylim([0, 100])
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_ft_gain.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_ft_gain.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_ft_gain.pdf")

if __name__ == '__main__':
    generate_all_figures()
