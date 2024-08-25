# Predictive-Modeling-of-Breast-Cancer-using-FNA-Data
This repository features a machine learning model for breast cancer classification using Fine Needle Aspiration (FNA) data. It includes data preprocessing, model implementation with advanced hyperparameter tuning, and performance evaluation using metrics like accuracy and AUC-ROC. 
The project leverages state-of-the-art techniques in deep learning, particularly Recurrent Neural Networks (RNN) with LSTM layers, to achieve high accuracy and robust predictions.

The contents include a Jupyter notebooks for implementation, a README file with detailed instructions, and an MIT license

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Figures](#figures)
- [License](#license)

## Project Overview
Breast cancer is a critical health concern worldwide, and early detection is key to improving survival rates. This project focuses on building a predictive model to classify breast cancer tumors as benign or malignant using data from fine needle aspiration (FNA) of breast masses. The goal is to create a highly accurate and efficient model that can assist in early diagnosis.

### Key Features:
- **Deep Learning Approach:** Utilizes Recurrent Neural Networks (RNN) with LSTM layers for handling sequential data.
- **Hyperparameter Optimization:** Employs Optuna for fine-tuning hyperparameters to maximize model performance.
- **Comprehensive Evaluation:** Includes multiple metrics such as accuracy, precision, recall, F1-score, and AUC-ROC for thorough evaluation.

## Dataset
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is publicly available and widely used in the field of breast cancer research.

### Dataset Details:
- **Source:** UCI Machine Learning Repository
- **Number of Instances:** 569
- **Number of Attributes:** 32 (including the target label)
- **Target Labels:** Benign (357 cases) and Malignant (212 cases)
- **Attributes:** Mean, standard error, and worst values for ten real-valued features are computed for each cell nucleus.

### Attributes:
- Radius (mean of distances from the center to points on the perimeter)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness (perimeter^2 / area - 1.0)
- Concavity (severity of concave portions of the contour)
- Concave points (number of concave portions of the contour)
- Symmetry
- Fractal dimension ("coastline approximation" - 1)

## Installation

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)
- Git (for cloning the repository)

## Model Architecture
The model is built using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers, which are effective in handling sequential data. The architecture includes:

- **Input Layer:** Accepts the input features.
- **LSTM Layers:** Captures the sequential dependencies in the data.
- **Dense Layers:** Fully connected layers for decision making.
- **Output Layer:** Provides the final classification (Benign or Malignant).

### Hyperparameter Tuning
Optuna is used for hyperparameter tuning, optimizing key parameters such as learning rate, LSTM units, batch size, and dropout rates to achieve the best model performance.

## Results
The model achieved the following performance metrics:
- **Accuracy:** 98.24%
- **AUC-ROC:** 99.80%
- **Precision, Recall, F1-score:** High scores across both classes, indicating balanced performance.

### Confusion Matrix
|               | Predicted Benign | Predicted Malignant |
|---------------|------------------|---------------------|
| Actual Benign | 70               | 1                   |
| Actual Malignant | 1             | 42                  |

## Figures
To provide further insight into the model’s performance, the following figures are included:

- **ROC Curve** – Illustrates the trade-off between the true positive rate and false positive rate.
- **Training and Validation Accuracy** – Depicts the accuracy of the model over epochs.
- **Training and Validation Loss**  – Represents the loss values over epochs.
- **Confusion Matrix ** – A heatmap visualizing the true positives, true negatives, false positives, and false negatives.

## License
This project is licensed under the MIT License. See the [LICENSE] file for more details.

---
