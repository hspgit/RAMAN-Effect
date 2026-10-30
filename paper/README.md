# Paper: residual vs. plain 1D CNNs on bacteria-ID

Full-page-width (single-column) IEEE template (`IEEEtran`) that follows the
agreed outline:
title, abstract, introduction, dataset, models, results (6.1–6.5), discussion,
conclusion, back matter and appendix.

## Layout

```
paper/
├── overleaf/            <- the Overleaf project: upload exactly this folder
│   ├── main.tex              (the paper; full-page single-column layout)
│   ├── references.bib
│   └── figures/
├── scripts/             <- figure generators; run here, never on Overleaf
│   └── make_fig_spectra.py   (writes into overleaf/figures/)
└── README.md
```

Only `overleaf/` goes to Overleaf. The scripts need `data/` and the Python
environment from this repo, so they stay here.

## New to Overleaf?

[`overleaf_guide.png`](overleaf_guide.png) is an annotated screenshot of
this project in Overleaf, with the 17 places you'll use numbered. Your first
ten minutes:

1. Open `main.tex` (3) and switch the editor to **Code** (4).
2. Change something small, then press **Recompile** (1) and watch the PDF (2).
3. Jump between sections with the **File outline** (6). Double-click the PDF
   to land on the matching line.
4. Search for `\TBD` and `\NOTE`: these are the red and blue text (13) still
   to be filled in.
5. To add a picture, upload it to `figures/` (7, 8) and paste the snippet at
   the bottom of the guide.

## Getting it into Overleaf

1. Zip the *contents* of `overleaf/`, so that `main.tex` sits at the top of
   the zip and not inside a folder.
2. In Overleaf: **New Project → Upload Project**, then choose the zip.
3. Open **Menu** and check that the compiler is **pdfLaTeX** and the main
   document is **main.tex**. Then click **Recompile**.

### Page layout

The paper uses the full page width: `\documentclass[journal,onecolumn]{IEEEtran}`.
If the journal asks for two columns at submission, delete `,onecolumn` from
that line. Nothing else needs to change.

Size normal figures with `width=\figwidth`. In this layout that is 60% of the
page, so plots keep readable font sizes. If you switch to two columns it
becomes exactly one column.

To update a figure later, re-run its script here and drag the new file from
`overleaf/figures/` into the `figures` folder in Overleaf (or use Overleaf's
Git or GitHub sync, if your plan has it).

Red `[...]` marks numbers still waiting on experiments (`\TBD`). Blue notes
(`\NOTE`) are writing instructions. Search for both and remove them before
submission. To switch to the conference layout, change the `\documentclass`
line to `\documentclass[conference]{IEEEtran}`.

Units use two plain macros, `\sunit{2}` for "2 s" and `1792.4\wn` for
"1792.4 cm⁻¹", so the file compiles anywhere. Overleaf also has a working
`siunitx` if you prefer it.

## Adding figures

Put files in `figures/`, then reference them without the folder or extension.
Every pattern below already appears in `main.tex`.

| You want | Environment | Width | Example in main.tex |
|---|---|---|---|
| Normal plot | `figure` | `\figwidth` | Fig. 1, `fig_spectra` |
| Full-page-width figure | `figure*` | `\textwidth` | Fig. 3, confusion matrices |
| Two small panels side by side | `figure` + 2× `\subfloat` | `0.48\figwidth` each | Fig. 5 |
| Two panels across the page | `figure*` + 2× `\subfloat` | `0.48\textwidth` each | Fig. 3 |
| Block diagram | `figure` + `tikzpicture` | — | Fig. 2 |
| Figure not made yet | `\figorplaceholder{name.pdf}{height}` | optional `[width]` | Figs. 4, 5 |

The minimal figure:

```latex
\begin{figure}[!t]
  \centering
  \includegraphics[width=\figwidth]{fig_spectra}
  \caption{Caption goes below the figure.}
  \label{fig:spectra}
\end{figure}
% ...and in the text, before the figure appears: \cref{fig:spectra}
```

Publishing rules the template follows:

- **Vector first.** Save plots as PDF (`fig.savefig("x.pdf")`). A raster
  image must be at least 300 dpi at printed size, so ≥1050 px wide for one
  column (3.5 in) and ≥2150 px for full width (7.16 in).
- **Draw at a fixed size.** Make matplotlib figures 3.5 in wide (normal) or
  7.16 in (page-wide), with 7–8 pt fonts. In this layout `\figwidth` is
  4.3 in, so normal figures scale by about 1.2× and their labels come out
  close to the 10 pt body text. They also fit a two-column journal layout
  unscaled.
  `scripts/make_fig_spectra.py` is the reference example. It also embeds
  TrueType fonts (`pdf.fonttype = 42`), which IEEE PDF eXpress checks for.
- **Captions** go below figures and above tables. Label every panel (a), (b).
- **`figure*` / `table*`** float only to the top of a page, and no earlier
  than the page after the one where they are declared.
- **Colour.** Figures must still read in greyscale, so use direct labels or
  line styles, not colour alone.

## Placeholders to replace

`figures/confusion_matrix_*.png` and `figures/loss_curves.png` are copies
of old exploratory images (a 5-epoch run, dark theme). They only show how to
include a raster image. Regenerate them as PDFs from the final runs.

## References

`references.bib` holds the works cited so far. Check each DOI, volume and
page range against the publisher's record before submission.
