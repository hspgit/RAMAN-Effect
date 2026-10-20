# RNN Architecture for Raman Spectroscopy

## Overview
The Recurrent Neural Network (RNN) architecture was introduced to test the hypothesis that Raman spectra can be effectively modeled as 1D sequential data (time-series). Unlike Convolutional Neural Networks (CNNs) that extract local spatial features, RNNs process the spectrum sequentially, carrying forward a "hidden state" or memory of previous wavenumbers.

## Network Design
The chosen architecture is a **Bidirectional Long Short-Term Memory (LSTM)** network.

### Key Components
1. **Input Representation**: The 1D Raman spectrum (e.g., 1000 wavenumbers) is reshaped into a sequence of length $L=1000$ where each step has a feature dimension of $1$ (the intensity).
2. **Bidirectional LSTM Layers**: 
   - We use a 2-layer LSTM with a hidden size of 128.
   - **Bidirectionality** is crucial: it processes the spectrum forwards (low to high wavenumbers) and backwards (high to low wavenumbers). This allows the network to have full contextual awareness of the entire spectrum before making a classification.
3. **Dropout Regularization**: A dropout rate of 0.3 is applied to prevent overfitting on the sequence features.
4. **Classification Head**: The final hidden states from both the forward and backward passes of the last LSTM layer are concatenated, forming a 256-dimensional feature vector. This vector is passed through a fully connected linear layer to output the logits for the 30 bacterial classes.

## Theoretical Motivation
The RNN serves as the sequential processing baseline in the comparative study. By contrasting its performance with spatial models (CNNs) and attention-based models (Transformers), we can determine the most effective inductive bias for analyzing Raman spectra.
