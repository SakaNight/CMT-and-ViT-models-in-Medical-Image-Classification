# 🧪 CMT & ViT Models in Medical Image Classification

This project presents an empirical evaluation of two state-of-the-art architectures in deep learning—**Convolutional Neural Networks Meet Transformers (CMT)** and **Vision Transformers (ViT)**—for chest X-ray image classification tasks.  
We benchmarked several model variants and attention mechanisms on medical datasets (NIH & COVID), analyzing performance trade-offs across accuracy, efficiency, and interpretability.

---

## 📌 Objectives

- Evaluate and compare **CMT**, **ViT-Tiny**, and **ViT-Small** for medical image classification.
- Investigate the impact of different attention mechanisms:  
  - 🧠 Standard Self-Attention  
  - 🌀 Sparse Attention  
  - 📍 Local Attention
- Measure performance in terms of:
  - Accuracy and convergence
  - Computational efficiency
  - Interpretability via attention maps

---

## 🗂️ Project Structure

```bash
├── data_loader.py                   # Custom PyTorch DataLoader
├── model_cmt.py                     # CMT architecture
├── model_vit_small.py              # ViT-Small architecture
├── train_evaluate_cmt.py           # Train and evaluate CMT
├── train_evaluate_vit.py           # Train and evaluate ViT variants
├── result_comparison_charts.py     # Plotting utility for results
├── test_models.py                  # Evaluation script
├── prepare_local_dataset_*         # Preprocessing NIH & COVID data
└── README.md
