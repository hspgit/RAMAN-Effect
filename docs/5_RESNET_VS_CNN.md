# Architecture Comparison: ResNet vs. Standard CNN for Raman Spectroscopy

When building a deep learning model to classify 1D signals like Raman spectra, Convolutional Neural Networks (CNNs) are the standard choice. However, the architecture defined in this project (`Raman1DCNN` in `src/model.py`) is technically a **1D Residual Network (ResNet)**, which is an advanced variant of a CNN. 

This document explains the differences between a standard CNN and a ResNet, and why the ResNet approach is vastly superior for Raman spectra classification.

---

## 1. Standard 1D CNNs
A standard 1D CNN processes the 1,000-point Raman spectrum by sliding filters (kernels) across the data to detect local patterns. 
- **Early layers** detect simple features like sharp peaks, edges, and valleys.
- **Deeper layers** combine these simple features into complex, abstract patterns (e.g., the relative distance between three specific peaks).

### The Problem with Deep CNNs
To classify complex bacterial species accurately, you need a deep network with many layers. However, standard CNNs suffer from the **Vanishing Gradient Problem**. 
As the network gets deeper:
1. The signal from the original spectrum gets repeatedly multiplied and altered, often causing the network to "forget" small but critical chemical peaks.
2. During training, the error gradient (used to update the model weights) becomes infinitesimally small by the time it reaches the early layers, halting the learning process entirely.

---

## 2. The ResNet Approach (Our Architecture)

A Residual Network (ResNet) solves the vanishing gradient problem by introducing **Skip Connections** (or shortcuts). 

In `src/model.py`, this is implemented via the `ResidualBlock1D` class. Instead of forcing the data to pass linearly through every single convolution, a skip connection takes the input of a block and adds it directly to the output of that block.

### Why ResNet is Better for Raman Spectra:
1. **Preservation of Raw Signal:** Raman peaks can be very subtle. Skip connections ensure that raw, un-transformed signal data can bypass convolution layers and be directly accessible deeper in the network.
2. **Deeper Training:** We can safely stack more layers (our model uses 4 deep layers with multiple blocks) without the gradients vanishing. This allows the model to learn highly complex, non-linear relationships between distant peaks.
3. **Identity Mapping:** If a specific convolution layer isn't useful for a certain bacteria classification, the network can easily learn to zero out its weights and just rely on the skip connection. A standard CNN would struggle to pass the data through unchanged.

---

## 3. Breakdown of our `Raman1DCNN`

Our architecture is a custom-built 1D ResNet tailored specifically for the 1,000-feature Raman dataset:

- **Initial Convolution (`conv1`):** A large kernel (size 7) slides over the data with a stride of 2. This quickly downsamples the data and captures broad, low-frequency trends.
- **Residual Layers (`layer1` to `layer4`):** Four stages of `ResidualBlock1D`. The number of filters (channels) doubles at each stage (16 → 32 → 64 → 128), while the spatial dimension (length of the sequence) is halved. This forces the network to learn increasingly abstract, dense representations of the chemical fingerprint.
- **Adaptive Average Pooling (`avgpool`):** Flattens the final feature maps into a fixed-size 1D vector, making the model robust to slight shifts in the signal.
- **Classifier (`classifier`):** A fully connected neural network with a Dropout layer (0.4) to prevent overfitting, ending in the final 30 output nodes for the 30 bacteria classes.
