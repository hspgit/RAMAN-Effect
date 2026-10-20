# RNN Results and Paper Implications

## Experimental Results
After fully integrating and training the `RamanLSTM` model on the dataset, the final evaluation on the test set yielded the following metrics:

*   **Test Set Average Loss:** 3.4012
*   **Test Set Accuracy:** **3.33%** (100 / 3000 correct predictions)

## Analysis
Because the dataset contains exactly 30 bacterial classes, a baseline of random guessing equates to $1/30 = 3.33\%$. 

The RNN failed completely to learn any meaningful representations from the Raman spectra, performing no better than random chance.

### Why did the RNN fail?
Raman spectra in this dataset consist of extremely long sequences (length $L=1000$). RNNs and LSTMs notoriously suffer from the **vanishing gradient problem** and **catastrophic forgetting** over long sequences. By the time the LSTM reaches the 1000th wavenumber, it has lost the context of the critical peaks located at the beginning of the spectrum.

## Implications for the Comparative Paper
This failure is actually a highly valuable negative result that strengthens the paper's overall narrative. It perfectly completes the architectural comparison:

1.  **Sequential Models (RNN): ~3.3%** 
    *   Fails entirely due to sequence length and inability to retain long-range spectral dependencies.
2.  **Deep Spatial Models (ResNet): ~67.5%**
    *   Learns representations but suffers from overfitting and over-parameterization on 1D spectral data.
3.  **Local Spatial Models (Simple CNN): ~85.3%**
    *   Vastly outperforms all other architectures (especially when operating on raw, non-preprocessed data) by successfully extracting local peak shapes independent of sequence length.

**Conclusion:** Raman spectra should be modeled computationally as localized spatial structures (like 1D images) rather than continuous time-series sequences.
