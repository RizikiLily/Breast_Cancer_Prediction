
# Breast Cancer Detection Project

## Overview
This project aims to develop a machine learning model to predict whether a cell in breast tissue is malignant or benign based on various features derived from digitized images of breast mass. The dataset used contains clinical measurements of breast cancer tumors, including attributes such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Tools and Technologies Used
- Python
- Jupyter Notebook
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Project Structure
- `Breast_Cancer_Detection.ipynb`: Jupyter Notebook containing the exploratory data analysis (EDA), preprocessing steps, model training (Support Vector Machine), hyperparameter optimization using GridSearchCV, and evaluation.
- `breast-cancer-wisconsin-data-data.csv`: File containing the dataset used in the project.
- `README.md`: This file, providing an overview of the project and instructions for running the notebook.

## Getting Started
To run the notebook and reproduce the results:
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Open `Breast_Cancer_Detection.ipynb` using Jupyter Notebook or any compatible platform.
4. Follow the instructions provided in the notebook for step-by-step execution.

## Results
- Two reports are generated:
    - **Before Optimization**: Evaluation metrics (accuracy, precision, recall, F1-score, etc.) before hyperparameter optimization.
    - **After Optimization**: Evaluation metrics after hyperparameter optimization using GridSearchCV.
- Visualizations are included to compare model performance and demonstrate the impact of hyperparameter tuning.

## Conclusion
The project demonstrates the effectiveness of Support Vector Machine (SVM) in classifying breast cancer tumors as malignant or benign. By optimizing hyperparameters using GridSearchCV, we were able to improve the model's performance, as evidenced by the comparison of evaluation metrics before and after optimization.

