# 🌱 Plant Disease Classification using Deep Learning & Explainable AI

A deep learning pipeline that classifies plant leaf diseases across 6 vegetable types using a custom CNN and transfer learning, with model interpretability powered by Explainable AI (XAI) techniques.

## 📌 Overview

This project builds and compares multiple deep learning models to classify plant leaf diseases from images, aiming to support early disease detection in precision agriculture. It goes beyond raw accuracy by using Explainable AI methods to visualize *why* the model makes its predictions — an important step toward trustworthy, real-world-deployable agricultural AI systems.

## 📊 Dataset

- **21 classes** across **6 vegetable types**: Bitter Gourd, Bottle Gourd, Cauliflower, Cucumber, Eggplant, and Tomato
- Each class represents either a specific disease (e.g., Downey mildew, Bacterial spot, Leaf curl virus) or a healthy/fresh leaf
- **12,786 total images**
- Dataset was reorganized and flattened into a clean category-wise folder structure for training

## 🧠 Models & Approach

| Model | Type | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Custom CNN | Built from scratch | 95.54% | 0.9601 | 0.9554 | 0.9553 |
| DenseNet201 | Transfer Learning | **98.98%** | 0.9907 | 0.9898 | 0.9897 |

- **Custom CNN**: A convolutional neural network built from scratch, trained with Automatic Mixed Precision (AMP) and early stopping for efficient training
- **Transfer Learning**: Fine-tuned pretrained models (DenseNet201, ResNeXt101) on the dataset, with DenseNet201 achieving the best overall performance

## 🔍 Explainable AI (XAI)

To understand and validate what the model "looks at" when making predictions, the following interpretability techniques were applied:

- **Grad-CAM** — highlights regions of the image most influential to the prediction
- **Grad-CAM++** — an enhanced version of Grad-CAM for finer-grained localization
- **Eigen-CAM** — a gradient-free visualization technique
- **LIME** — perturbation-based local explanation of individual predictions

These visualizations help confirm the model is focusing on disease-relevant regions of the leaf rather than background artifacts, which is critical for building trust in agricultural AI applications.

## 🛠️ Tech Stack

- **Python**, **PyTorch**, **Torchvision**, **timm**
- **scikit-learn** (evaluation metrics)
- **pytorch-grad-cam**, **LIME** (explainability)
- **Matplotlib** (visualization)

## 📈 Key Takeaways

- Transfer learning (DenseNet201) significantly outperformed the custom CNN, improving accuracy from 95.54% to 98.98%
- XAI visualizations confirmed the model's predictions were grounded in biologically relevant leaf features, not spurious correlations
- The pipeline demonstrates an end-to-end workflow: dataset preprocessing → model training → evaluation → interpretability

*This project was completed as part of the Artificial Intelligence (CSE366) coursework at East West University.*
