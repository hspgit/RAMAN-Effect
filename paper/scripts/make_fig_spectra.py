"""Fig. 1 of the paper: one low-SNR spectrum against its isolate mean, and the
mean spectra of three isolates. Writes a vector PDF sized for one IEEE column.

    python paper/scripts/make_fig_spectra.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
# Lands in the Overleaf project folder; upload or sync it from there.
OUT = os.path.join(ROOT, "paper", "overleaf", "figures", "fig_spectra.pdf")

# Isolate indices and names from csho33/bacteria-ID config.py (STRAINS).
ISOLATES = [(3, "E. coli 1"), (17, "MRSA 2"), (0, "C. albicans")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz categorical slots 1-3
GREY = "#8a8a85"
INK = "#30302e"

# IEEE single column is 3.5 in wide; 8 pt text matches the 10 pt body after
# the figure is placed at \columnwidth without scaling.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # embed TrueType so publishers' font checks pass
})


def main():
    X = np.load(os.path.join(DATA, "X_reference.npy"), mmap_mode="r")
    y = np.load(os.path.join(DATA, "y_reference.npy"))
    w = np.load(os.path.join(DATA, "wavenumbers.npy"))

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.4]})

    idx, name = ISOLATES[0]
    rows = np.flatnonzero(y == idx)
    ax_a.plot(w, X[rows[0]], color=GREY, lw=0.8, label="Single spectrum", solid_joinstyle="round")
    ax_a.plot(w, X[rows].mean(axis=0), color=COLORS[0], lw=1.5,
              solid_joinstyle="round", label="Isolate mean")
    ax_a.set_ylabel("Norm. intensity")
    # The spectrum fills the whole panel, so the legend sits above it.
    ax_a.legend(frameon=False, fontsize=7, loc="lower right", ncol=2,
                bbox_to_anchor=(1.0, 0.98), borderaxespad=0, handlelength=1.5)
    ax_a.text(0.0, 1.04, "(a)", transform=ax_a.transAxes, va="bottom",
              color=INK)

    offset = 0.0
    for (idx, name), color in zip(ISOLATES, COLORS):
        mean = X[y == idx].mean(axis=0) + offset
        ax_b.plot(w, mean, color=color, lw=1.5, solid_joinstyle="round")
        ax_b.text(w.max() + 15, mean[np.argmax(w)], name, color=INK,
                  va="center", ha="left", fontsize=7, style="italic")
        offset += 0.6
    ax_b.set_ylabel("Mean intensity (offset)")
    ax_b.set_yticks([])
    ax_b.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax_b.text(0.0, 1.02, "(b)", transform=ax_b.transAxes, va="bottom",
              color=INK)
    ax_b.set_xlim(w.min(), w.max())

    for ax in (ax_a, ax_b):
        ax.grid(axis="x", color="#e4e3dd", lw=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout(h_pad=0.4)
    fig.subplots_adjust(right=0.78)  # room for the direct labels in (b)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
