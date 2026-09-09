# Data Exploration: Raman Spectra Dataset

This document provides an overview of the dataset used in the `RAMAN-Effect` project, including what the data represents in simple terms, exploratory data analysis (EDA) techniques, dataset statistics, and code snippets for visualizing the data.

## 1. What does this data actually represent? (A Layman's Guide)

If you aren't a chemist or physicist, "Raman spectroscopy" might sound intimidating, but the core concept is quite simple. 

Think of Raman spectroscopy as a **"chemical fingerprint scanner."** 

Here is how it works:
1. **Shooting a Laser:** A scientist shoots a laser beam (a single color of light) at a sample—in this case, different species of bacteria.
2. **The "Jiggle":** When the light hits the molecules in the bacteria, most of the light bounces off normally. However, a tiny fraction of the light hits the molecules and causes them to vibrate or "jiggle." 
3. **The Color Shift:** Because some of the light's energy is absorbed to make the molecules jiggle, the light that bounces back actually changes color (shifts in energy) ever so slightly. 
4. **The Fingerprint:** Different chemical bonds (like carbon-oxygen or carbon-hydrogen bonds) jiggle in very specific, predictable ways. By measuring *exactly* how much the light changed energy, we can map out a "spectrum" (a line graph) showing the unique vibrations.

**An Everyday Example:**
Imagine throwing a tennis ball at a set of tuning forks or guitar strings. Depending on how thick or tight the strings are, they will ring out with different, specific musical notes when hit. 
In Raman spectroscopy:
- The **laser** is the tennis ball.
- The **bacteria's molecules** (proteins, DNA, fats) are the guitar strings. 
- The **Raman spectrum** (our 1,000 features) is the microphone recording the unique chord that rings out when the laser hits. E. coli might "sound" like a C-major chord, while Staphylococcus might "sound" like a G-minor chord!

**In this dataset:**
- A **sample** (one row of data) is the graph of these light shifts for a specific bacterial cell.
- The **1,000 features** (the length of our data array) represent 1,000 different specific energy shifts we are measuring. The number at each point is the *intensity*—basically, how strongly the light shifted at that exact energy level.
- By looking at the pattern of these 1,000 numbers (peaks and valleys), our AI model can learn to recognize the unique "chemical fingerprint" of 30 different types of bacteria, effectively acting as a high-tech, microscopic barcode scanner!

**A Concrete Data Sample:**
If you open the dataset and look at the very first bacteria sample (Row 0), here is exactly what the data looks like:

**The Features (`X_reference.npy`):**
```python
X[0] = [0.3552, 0.3070, 0.3431, 0.3399, 0.3479, 0.3869, ... (994 more numbers)]
```
*These 1,000 numbers are the intensities of the "chord" mentioned above. They have already been normalized to sit between 0 and 1. If plotted on a graph, these numbers form the peaks and valleys of the spectrum.*

**The Label (`y_reference.npy`):**
```python
y[0] = 0.0
```
*This single number is the target class (the bacteria species ID). In this case, `0.0` represents the very first species of bacteria in our 30-class list.*

---

## 2. Dataset Overview

The main training dataset is stored in `.npy` (NumPy) files under the `data/` directory.

- **Features (`X_reference.npy`)**: Contains the Raman spectra intensities (the fingerprint graphs).
- **Labels (`y_reference.npy`)**: Contains the target classes (the 30 different bacteria species IDs).

### Basic Statistics
Based on a preliminary load of the reference data, the dataset exhibits the following properties:
- **Total Samples**: 60,000
- **Spectral Length (Features)**: 1,000 points per spectrum
- **Number of Classes**: 30 distinct classes
- **Class Distribution**: Perfectly balanced! There are exactly 2,000 samples for each of the 30 classes.

*Note: This data structure closely mirrors the 30-class Bacteria isolate dataset available via `ramanspy`.*

---

## 3. Loading the Data

You can easily load and inspect the shape of the data using `numpy`:

```python
import numpy as np

# Load data
X = np.load('data/X_reference.npy')
y = np.load('data/y_reference.npy')

print(f"Features shape: {X.shape}") # Expected: (60000, 1000)
print(f"Labels shape: {y.shape}")   # Expected: (60000,)
```

---

## 4. Visualizing Spectra

Because Raman data is highly dimensional (1D arrays of intensities across different wavenumbers), plotting the spectra is the best way to understand the variance and signature of different classes.

### Option A: Using `matplotlib` (Raw Data)
Here is a quick script to plot a few random samples from the dataset to see what the raw spectra look like:

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.load('data/X_reference.npy')
y = np.load('data/y_reference.npy')

plt.figure(figsize=(10, 5))

# Plot one sample from 3 different classes
for class_id in [0, 1, 2]:
    # Find the first index of the current class
    idx = np.where(y == class_id)[0][0]
    plt.plot(X[idx], label=f'Class {class_id}')

plt.title('Raw Raman Spectra Samples')
plt.xlabel('Raman Shift (Features / Index)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
```

### Option B: Using `ramanspy` (Advanced)

Our project uses `ramanspy` for robust preprocessing. We can also use it to visualize the mean spectra across different species. 

```python
import numpy as np
import ramanspy
import matplotlib.pyplot as plt

# Load data
X = np.load('data/X_reference.npy')
y = np.load('data/y_reference.npy')

# Group spectra by class
unique_classes = np.unique(y)
spectra_by_class = [[X[y == cls]] for cls in unique_classes]

# Apply normalization to make them comparable
pipeline = ramanspy.preprocessing.normalise.MinMax()
normalized_spectra = pipeline.apply(spectra_by_class)

# Plot the mean spectra of a subset of classes (e.g., first 5 classes)
plt.figure(figsize=(8, 10))
ramanspy.plot.mean_spectra(
    normalized_spectra[:5], 
    label=[f"Class {i}" for i in range(5)], 
    plot_type="single stacked"
)
plt.title("Mean Spectra Comparison (Classes 0-4)")
plt.savefig("results/mean_spectra_exploration.png", bbox_inches='tight')
plt.show()
```

---

## 5. Preprocessing Impact

Raw Raman spectra often suffer from noise, cosmic ray spikes, and baseline drift (fluorescence). Before feeding the data to the 1D CNN, we apply a pipeline in `src/dataset.py`:

1. **Despiking** (`WhitakerHayes`): Removes sharp, non-Raman spikes.
2. **Denoising** (`SavGol`): Smooths the signal using a Savitzky-Golay filter.
3. **Baseline Correction** (`ASPLS`): Flattens the underlying fluorescence baseline.
4. **Normalisation** (`MinMax`): Scales the intensities to a standard range [0, 1].

**Exploration Tip:** When exploring the data, it is highly recommended to plot a single spectrum *before* and *after* passing it through this `ramanspy` pipeline to visually verify the signal-to-noise ratio improvements.
