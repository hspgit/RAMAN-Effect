# Setup Guide for RAMAN-Effect

This document provides step-by-step instructions on how to set up the environment, configure the data, and run the `test.py` script.

## 1. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

1. **Create a Virtual Environment:**
   Run the following command in your terminal:
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the Virtual Environment:**
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Install Dependencies:**
   Install the required libraries from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Bugfix for `ramanspy` Library

The current version of the `ramanspy` library uses a deprecated matplotlib function which will cause an error when generating plots. You need to apply a small patch to the installed library.

1. Open the file `.venv/lib/python3.13/site-packages/ramanspy/plot/_core.py` (your Python version may vary slightly, e.g. `python3.12`).
2. Navigate to around line 175.
3. Replace the following line:
   ```python
   cmap = plt.cm.get_cmap()  # using matplotlib's default colormap
   ```
   With:
   ```python
   cmap = plt.get_cmap()  # using matplotlib's default colormap
   ```

## 3. Data Setup

The `test.py` script uses the Raman Bacteria Dataset (based on the work of Ho, CS. et al. (2019)). This dataset must be manually downloaded before you can use the `ramanspy` loading functions.

1. Ensure there is a folder named `data` in the root directory of the project.
2. Download the data from the authors' [DropBox repository](https://www.dropbox.com/sh/gmgduvzyl5tken6/AABtSWXWPjoUBkKyC2e7Ag6Da?dl=0).
3. Place all the downloaded `.npy` files for the dataset splits into the `data` directory. Specifically, the script expects data required for validation (`X_finetune.npy`, `y_finetune.npy`, `X_reference.npy`, etc.).

Once downloaded, the `ramanspy` library function `ramanspy.datasets.bacteria()` will be able to load the data successfully.

## 4. Running the Test Script

Once the environment and data are set up, you can run the provided script.

1. Execute the script:
   ```bash
   python test.py
   ```
2. The script will load the bacteria data, normalize the spectra, generate a single stacked plot of the mean spectra for each species, and save it as `plot.png` in your current directory. It will also display the plot interactively.
