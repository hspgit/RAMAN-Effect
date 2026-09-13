# Training Strategy & Regularization

Building a powerful ResNet is only half the battle; how you train it dictates whether it successfully converges or simply memorizes the noise (overfitting). In addition to Data Augmentation, the `RAMAN-Effect` pipeline utilizes a specific set of optimization and regularization strategies defined in `main.py`.

## 1. Optimizer and Weight Decay (L2 Regularization)
We use the **Adam optimizer** (`optim.Adam`), which adapts the learning rate for each parameter individually. 

To further combat overfitting (alongside Dropout and Augmentation), we apply **Weight Decay (`weight_decay=0.01`)**. 
- **What it does:** It adds a penalty to the loss function proportional to the size of the model's weights.
- **Why it is needed:** Deep networks can assign massive weights to specific, noisy features (like a single pixel or a specific wavenumber spike) to cheat during training. Weight decay mathematically forces the model to keep its weights small and distributed evenly, encouraging it to look at the *entire* Raman spectrum rather than hyper-focusing on noise.

## 2. Cosine Annealing Learning Rate Scheduler
Instead of keeping a static learning rate or stepping it down rigidly, we use `optim.lr_scheduler.CosineAnnealingLR`.

- **What it does:** It starts with a high learning rate and gradually decays it following the curve of a cosine wave down to near zero by the final epoch.
- **Why it is needed:** Early in training, a high learning rate allows the model to make large leaps and quickly find the general area of the optimal solution. As epochs progress, the learning rate smoothly shrinks, allowing the model to make finer, more precise adjustments without overshooting the optimal weights. This smooth descent prevents the validation loss from thrashing.

## 3. Two-Phase Training (Pre-training & Fine-tuning)
The dataset is split across different conditions (e.g., highly controlled reference lab data vs. messy clinical data). To handle this domain shift, we use a two-phase training approach:

1. **Phase 1: Pre-training (Reference Data):** The model is trained on the large, balanced `X_reference.npy` dataset. The goal here is for the model to learn the general shape of all 30 bacterial classes. It uses the standard learning rate (e.g., `lr=0.001`) and the Cosine Annealing scheduler.
2. **Phase 2: Fine-tuning:** The model is then exposed to the `X_finetune.npy` dataset. Crucially, we drop the learning rate by a factor of 10 (`lr * 0.1`). 
   - **Why reduce the LR?** We don't want to destroy the deep, foundational chemical representations the model learned during Phase 1. The low learning rate allows the model to gently "tweak" its final layers to adapt to the specific idiosyncrasies of the new fine-tuning data, leading to a much higher score on the final unseen test set.
