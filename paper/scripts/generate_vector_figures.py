import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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


def load_canonical_runs(repo_root, pattern):
    out = {}
    for d in sorted(glob.glob(os.path.join(repo_root, pattern))):
        man = json.load(open(os.path.join(d, 'manifest.json')))
        met = json.load(open(os.path.join(d, 'metrics.json')))
        seed = man['args']['seed']
        yt = np.load(os.path.join(d, 'test_y_true.npy'))
        yp = np.load(os.path.join(d, 'test_y_pred.npy'))
        out[seed] = {'dir': d, 'man': man, 'met': met, 'yt': yt, 'yp': yp}
    return out


def generate_confusion_matrices():
    print("Generating 3-seed 30-class confusion matrix vector PDFs...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(repo_root, 'paper', 'overleaf', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    cnn_runs = load_canonical_runs(repo_root, 'runs/*legacy_cnn_abl-minmax_only*')
    res_runs = load_canonical_runs(repo_root, 'runs/*resnet_abl-minmax_only*')

    def mean_cm_and_acc(runs_dict):
        cms = []
        accs = []
        for s, g in sorted(runs_dict.items()):
            cms.append(confusion_matrix(g['yt'], g['yp'], labels=np.arange(30)))
            accs.append((g['yt'] == g['yp']).mean() * 100.0)
        return np.mean(cms, axis=0), float(np.mean(accs)), float(np.std(accs, ddof=1))

    cm_plain, acc_plain, sd_plain = mean_cm_and_acc(cnn_runs)
    cm_resnet, acc_resnet, sd_resnet = mean_cm_and_acc(res_runs)

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

    for cm, title, acc, sd, filename in [
        (cm_plain, 'Plain 1D CNN', acc_plain, sd_plain, 'confusion_matrix_legacy_cnn'),
        (cm_resnet, '1D ResNet', acc_resnet, sd_resnet, 'confusion_matrix_resnet')
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=300)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.set_ylabel('Mean Test Count (per 100)', rotation=-90, va="bottom", fontsize=7.5)

        ax.set(xticks=np.arange(30),
               yticks=np.arange(30),
               xticklabels=class_names,
               yticklabels=class_names)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        ax.set_xlabel('Predicted Isolate', fontweight='bold', fontsize=8)
        ax.set_ylabel('True Isolate', fontweight='bold', fontsize=8)
        ax.set_title(f'{title} Confusion Matrix (Acc: {acc:.1f} \u00b1 {sd:.1f}%)',
                     fontweight='bold', fontsize=8.5, pad=6)

        fig.tight_layout()
        pdf_path = os.path.join(output_dir, f'{filename}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {pdf_path}")


def generate_loss_curves():
    print("Generating 3-seed training and validation loss curves vector PDF...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(repo_root, 'paper', 'overleaf', 'figures')

    cnn_runs = load_canonical_runs(repo_root, 'runs/*legacy_cnn_abl-minmax_only*')
    res_runs = load_canonical_runs(repo_root, 'runs/*resnet_abl-minmax_only*')

    def collect_curves(runs_dict):
        tr_mat = np.full((len(runs_dict), 100), np.nan)
        va_mat = np.full((len(runs_dict), 100), np.nan)
        for idx, (s, g) in enumerate(sorted(runs_dict.items())):
            pre = [h for h in g['met']['history'] if h.get('phase') == 'pretrain']
            for h in pre:
                ep = int(h['epoch']) - 1
                if 0 <= ep < 100:
                    tr_mat[idx, ep] = h['train_loss']
                    va_mat[idx, ep] = h['val_loss']
            # Forward-fill after early stopping so the 3-seed curve represents active best state
            last_valid = np.where(~np.isnan(tr_mat[idx]))[0][-1]
            if last_valid < 99:
                tr_mat[idx, last_valid + 1:] = tr_mat[idx, last_valid]
                va_mat[idx, last_valid + 1:] = va_mat[idx, last_valid]
        return tr_mat, va_mat

    c_tr, c_va = collect_curves(cnn_runs)
    r_tr, r_va = collect_curves(res_runs)
    epochs = np.arange(1, 101)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 6.8,
        'lines.linewidth': 1.5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=300)

    # Plain CNN train & val
    ax.plot(epochs, np.mean(c_tr, axis=0), color='#1f77b4', linestyle='-',
            label='Plain CNN (Train)')
    ax.plot(epochs, np.mean(c_va, axis=0), color='#1f77b4', linestyle='--',
            label='Plain CNN (Val)')
    ax.fill_between(epochs,
                    np.mean(c_va, axis=0) - np.std(c_va, axis=0),
                    np.mean(c_va, axis=0) + np.std(c_va, axis=0),
                    color='#1f77b4', alpha=0.14)

    # 1D ResNet train & val
    ax.plot(epochs, np.mean(r_tr, axis=0), color='#d62728', linestyle='-',
            label='1D ResNet (Train)')
    ax.plot(epochs, np.mean(r_va, axis=0), color='#d62728', linestyle='--',
            label='1D ResNet (Val)')
    ax.fill_between(epochs,
                    np.mean(r_va, axis=0) - np.std(r_va, axis=0),
                    np.mean(r_va, axis=0) + np.std(r_va, axis=0),
                    color='#d62728', alpha=0.14)

    ax.set_xlabel('Pre-training Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Pre-training & Validation Loss (3 Seeds)', fontweight='bold')
    ax.set_xlim([1, 100])
    ax.set_ylim([0.15, 1.55])
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, ncol=1)
    ax.grid(True)
    fig.tight_layout()

    pdf_path = os.path.join(output_dir, 'loss_curves.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {pdf_path}")


if __name__ == '__main__':
    generate_confusion_matrices()
    generate_loss_curves()
