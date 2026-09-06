# Raman Spectra 1D-ResNet Training Summary

This document summarizes the training run and evaluation results of the updated 1D-ResNet deep learning model for classifying 30 species of bacteria based on Raman spectra.

## Training Configuration
* **Command Executed:** `python3 train.py --epochs 10 --finetune-epochs 3 --batch-size 32`
* **Hardware Device:** `mps` (Apple Metal Performance Shaders)
* **Preprocessing:** Handled via `ramanspy` (despiking, Savitzky-Golay denoising, ASPLS baseline correction, MinMax normalization)
* **Classes Detected:** 30

## 1. Reference Pre-training Results
The model was pre-trained on the large reference dataset (`X_reference.npy`, 60,000 samples) for 10 epochs to learn general biochemical features.

| Epoch | Final Loss | Final Accuracy |
| :--- | :--- | :--- |
| Epoch 1 | 0.644 | 48.28% |
| Epoch 2 | 0.767 | 71.36% |
| Epoch 3 | 0.909 | 76.58% |
| Epoch 4 | 0.434 | 79.35% |
| Epoch 5 | 0.885 | 81.67% |
| Epoch 6 | 0.295 | 83.48% |
| Epoch 7 | 0.281 | 85.28% |
| Epoch 8 | 0.343 | 86.81% |
| Epoch 9 | 0.378 | 88.17% |
| **Epoch 10** | **0.322** | **89.09%** |

*Checkpoint Saved:* `models/raman_reference_model.pth`

## 2. Fine-tuning Results
The model was fine-tuned on the specialized dataset (`X_finetune.npy`, 3,000 samples) for 3 epochs to adapt to domain shifts.

| Epoch | Loss | Accuracy |
| :--- | :--- | :--- |
| Epoch 1 | 2.579 | 37.50% |
| Epoch 2 | 1.695 | 59.38% |
| **Epoch 3** | **1.177** | **68.75%** |

*Checkpoint Saved:* `models/raman_finetuned_model.pth`

## 3. Final Evaluation
The fine-tuned model was evaluated on the unseen test dataset (`X_test.npy`, 3,000 samples).

* **Test Average Loss:** `1.0177`
* **Test Accuracy:** `67.83%` (2035 / 3000 correct)

## Conclusion & Next Steps
The architectural upgrade to a 1D-ResNet massively improved the model's capacity, driving reference training accuracy to **89%**. Furthermore, fine-tuning enabled the model to reach **67.8% accuracy** on the test set, an absolute improvement of ~30% over the previous shallow CNN architecture.

To close the domain gap further and push test accuracy above 80%, future training runs should consider increasing the fine-tuning epochs (e.g., `--finetune-epochs 15`), as the loss was still decreasing rapidly by Epoch 3.
