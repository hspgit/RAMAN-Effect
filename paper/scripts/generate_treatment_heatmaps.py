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

ATCC_GROUPINGS = {
    3: 0, 4: 0, 9: 0, 10: 0, 2: 0, 8: 0, 11: 0, 22: 0,
    19: 1,
    12: 2, 13: 2,
    14: 3, 18: 3, 15: 3, 16: 3, 17: 3, 20: 3, 21: 3,
    24: 4, 23: 4,
    26: 5, 27: 5, 28: 5, 29: 5, 25: 5, 6: 5, 7: 5,
    5: 6,
    0: 7, 1: 7
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


def main():
    print("Evaluating 3-seed canonical runs for Empiric Antibiotic Treatment Group Heatmaps...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(repo_root, 'paper', 'overleaf', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    cnn_runs = load_canonical_runs(repo_root, 'runs/*legacy_cnn_abl-minmax_only*')
    res_runs = load_canonical_runs(repo_root, 'runs/*resnet_abl-minmax_only*')

    def compute_mean_treatment_cm(runs_dict):
        cms = []
        accs = []
        for s, g in sorted(runs_dict.items()):
            y_grp = np.array([ATCC_GROUPINGS[int(i)] for i in g['yt']])
            p_grp = np.array([ATCC_GROUPINGS[int(i)] for i in g['yp']])
            cms.append(confusion_matrix(y_grp, p_grp, labels=np.arange(8)))
            accs.append((y_grp == p_grp).mean() * 100.0)
        mean_cm = np.mean(cms, axis=0)
        norm_cm = mean_cm / mean_cm.sum(axis=1)[:, np.newaxis] * 100.0
        return mean_cm, norm_cm, float(np.mean(accs)), float(np.std(accs, ddof=1))

    cm_plain, cm_plain_norm, acc_plain, sd_plain = compute_mean_treatment_cm(cnn_runs)
    cm_resnet, cm_resnet_norm, acc_resnet, sd_resnet = compute_mean_treatment_cm(res_runs)

    short_labels = ["MEM", "CIP", "TZP", "VAN", "CRO", "PEN", "DAP", "CAS"]

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
    })

    def plot_single_cm(cm_raw, cm_norm, title, acc_total, sd_total, filename):
        fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=8)

        ax.set(xticks=np.arange(8),
               yticks=np.arange(8),
               xticklabels=short_labels,
               yticklabels=short_labels)

        ax.set_xlabel('Predicted Treatment Group', fontweight='bold', fontsize=8.5)
        ax.set_ylabel('Ground Truth Treatment Group', fontweight='bold', fontsize=8.5)
        ax.set_title(f'{title} (Overall: {acc_total:.1f} \u00b1 {sd_total:.1f}%)',
                     fontsize=9, fontweight='bold', pad=8)

        thresh = 50.0
        for i in range(8):
            for j in range(8):
                val = cm_norm[i, j]
                count = cm_raw[i, j]
                if val >= 0.5:
                    text_str = f"{val:.1f}%\n({count:.1f})"
                else:
                    text_str = f"{count:.1f}" if count >= 0.1 else "0"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text_str,
                        ha="center", va="center",
                        color=color, fontsize=5.6,
                        fontweight='semibold' if i == j else 'normal')

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}.pdf")

    plot_single_cm(cm_plain, cm_plain_norm, 'Plain 1D CNN', acc_plain, sd_plain, 'heatmap_treatment_plain')
    plot_single_cm(cm_resnet, cm_resnet_norm, '1D ResNet', acc_resnet, sd_resnet, 'heatmap_treatment_resnet')


if __name__ == '__main__':
    main()
