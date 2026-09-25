# Week 2 Summary: Raman Spectra Classification Optimization

This folder contains a summary of the work done during Week 2, focused on model optimization, evaluation, and visualization for the 1D Raman spectra classification task (30 bacterial classes).

## Contents

1. **[Model Experiments](model_experiments.md)**: Details our evaluation of various deep learning architectures (Legacy CNN, ResNet, Multiscale 1D-CNN, Transformers, and FusionNet).
2. **[Evaluation & Validation](evaluation.md)**: Outlines our robust verification processes, including K-Fold Cross Validation and visualization using Confusion Matrix heatmaps.
3. **[Interactive Dashboard](dashboard.md)**: Highlights the Streamlit web application built to visualize and compare training logs and metrics.

## Key Outcomes
* Discovered that **Legacy CNN** is the most effective architecture for this dataset, achieving **76.50% test accuracy**. 
* Proven that shallower models with data augmentation outperform deep and complex architectures (like Transformers and ResNets) due to the simplicity of the 1D spatial data.
* Set up a strong, reproducible training and evaluation pipeline with cross-validation and interactive dashboards.
