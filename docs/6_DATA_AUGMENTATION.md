# Data Augmentation for Raman Spectra

Deep learning models like our 1D ResNet require large amounts of diverse data to train effectively without overfitting. While our dataset has 60,000 samples, we can artificially expand this diversity by applying **data augmentation**—slightly modifying the existing training data on the fly at each epoch.

In this project, data augmentation is implemented in `src/dataset.py` via the `AugmentedDataset` PyTorch wrapper. We apply three specific transformations that simulate real-world physical and instrumental variations in Raman spectroscopy.

---

## 1. Random Shift (Roll)
**Implementation:** `np.roll(x_np, shift, axis=-1)` where shift is between -5 and 5 indices.

- **What it does:** Shifts the entire spectrum slightly to the left or right along the wavenumber axis. Because we use `np.roll`, data that falls off one end wraps around to the other.
- **Why it is needed:** Raman spectrometers often suffer from minor miscalibrations due to temperature changes in the room, physical bumps to the instrument, or slight optical misalignment. This causes the peaks to shift slightly from one day to the next. By artificially shifting the data during training, we force the CNN to recognize the *pattern* and relative distances of the peaks, rather than memorizing their exact index positions.

## 2. Random Gaussian Noise
**Implementation:** `np.random.normal(0, 0.01, x_np.shape)`

- **What it does:** Injects a small amount of random white noise (Gaussian noise with a mean of 0 and a standard deviation of 0.01) into every point on the spectrum.
- **Why it is needed:** Every electronic sensor has baseline thermal and electronic noise (often called "dark noise" or "shot noise"). The amount of noise can vary based on the exposure time and the quality of the sensor. Adding artificial noise prevents the model from overfitting to the clean, highly-processed signal and makes it more robust when deployed in real-world scenarios with lower-quality sensors.

## 3. Random Scale (Multiplicative)
**Implementation:** `x_np * scale` where scale is between 0.95 and 1.05.

- **What it does:** Multiplies the entire spectrum's intensity by a random scalar value, effectively increasing or decreasing the height of all peaks by up to 5%.
- **Why it is needed:** The overall intensity of a Raman signal is rarely consistent. It depends heavily on:
  - **Laser Power:** Slight fluctuations in the laser's output.
  - **Sample Concentration:** How dense the bacteria sample is.
  - **Focus:** How perfectly the microscope objective is focused on the sample.
  
  Scaling the data artificially teaches the model that a 5% taller version of a spectrum belongs to the exact same bacterial class, forcing the model to learn the *shape* and *relative heights* of the peaks rather than absolute intensity values.

---

## Conclusion
By applying these three augmentations randomly and dynamically during the training loop, the model essentially sees a uniquely deformed, realistic version of the data at every epoch. This drastically reduces overfitting and prepares the model to handle messy, real-world clinical data.
