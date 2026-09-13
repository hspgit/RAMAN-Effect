# Legacy CNN Baseline: Training Summary

This document summarizes the training run for the Legacy CNN model (originally from `NormalizedModelWithBootstrapping.ipynb`), implemented in PyTorch and run on the `RAMAN-Effect` dataset.

## 1. Run Configuration
* **Model**: Legacy 1D CNN (6 Convolutional layers with Average Pooling and Batch Normalization)
* **Script**: `legacy_baseline/train.py`
* **Device**: MPS (Apple Silicon GPU)
* **Dataset**: 60,000 samples (`X_reference.npy`) preprocessed with `RamanSPy`
* **Data Split**: 40,200 training samples (67%) / 19,800 validation samples (33%)
* **Hyperparameters**: 
  * Optimizer: Adam (lr=2e-4, weight_decay=0.01)
  * Batch Size: 128
  * Epochs: 10

## 2. Training Progression

The model showed steady learning over the 10 epochs. Below is a summary of the training phase:

* **Epoch 1**: The model started learning, ending the epoch with a Training Loss of `1.7972` and Accuracy of `31.06%`.
* **Epoch 3**: Substantial improvement was seen, crossing the 60% threshold (`64.84%` Training Accuracy).
* **Epoch 5 (Midpoint)**: 
  * **Training Accuracy**: `73.40%` (Loss: `1.0034`)
  * **Validation Accuracy**: `73.87%` (Loss: `0.9082`)
* **Epoch 10 (Final)**:
  * **Training Accuracy**: `79.28%` (Loss: `0.7762`)

## 3. Final Evaluation

After 10 epochs, the model was evaluated on the unseen validation/test set of 19,800 samples:

* **Final Test Accuracy**: **78.70%** (15,582 correct predictions out of 19,800)
* **Final Test Loss**: **0.7349**

## 4. Key Takeaways

1. **Convergence is Healthy**: The training and validation accuracies are very close (`79.28%` vs `78.70%`). This indicates that the model is generalizing well and is not currently overfitting.
2. **Room for Improvement**: While 78.7% is a solid baseline after only 10 epochs, this sets the benchmark for the newer **ResNet** model. 
3. **Next Steps**: To prove the ResNet is a superior architecture, the ResNet should be trained for 10 epochs using the same `RamanSPy` preprocessing pipeline and compared against this **78.70%** accuracy target.
