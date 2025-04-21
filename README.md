# Neural Network Employee Analysis

## Overview

This project aims to develop a neural network for HR to predict employee attrition and identify suitable departments for employees.  The model utilizes a branched neural network to predict these two key factors.

## Repository Structure

*   `attrition.ipynb`: Jupyter Notebook containing the neural network model, data preprocessing, and analysis.

## Objective

The primary goals of this project are:

*   Predict employee attrition using relevant features.
*   Determine the most suitable department for each employee based on their characteristics.

## Data

The dataset used in this project contains information on employee demographics, job satisfaction, and other factors relevant to attrition and department suitability.

## Model Architecture

The neural network architecture consists of:

*   Shared layers:  `shared1 (Dense)` and `shared2 (Dense)`
*   Two branched outputs: one for attrition prediction and another for department prediction.
*   Attrition branch: `attrition_hidden1 (Dense)`, `attrition_hidden2 (Dense)`, and `attrition_output (Dense)`
*   Department branch: `department_hidden1 (Dense)`, `department_hidden2 (Dense)`, and `department_output (Dense)`

## Dependencies

*   Python 3.6 or higher
*   Libraries:
    *   TensorFlow
    *   Scikit-learn
    *   Pandas
    *   NumPy

## Usage

1.  Clone the repository: `git clone [repository URL]`
2.  Open `attrition.ipynb` in Google Colab or Jupyter Notebook.
3.  Execute the cells in sequence to preprocess the data, train the model, and evaluate the results.

## Results

The model was trained, and the final evaluation metrics are:

*   Attrition Prediction Accuracy: 82.61%
*   Department Prediction Accuracy: 65.22%

These accuracy scores are calculated in the notebook using the following lines of code:

```
print("Attrition Accuracy: %.2f%%" % (results * 100))
print("Department Accuracy: %.2f%%" % (results * 100))
```

## Next Steps

*   Further hyperparameter tuning to improve model performance.
*   Exploring additional features that may contribute to prediction accuracy.
