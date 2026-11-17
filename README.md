# Task 39: Confusion Matrix - Titanic Survival Classification

## Overview
This project focuses on generating and interpreting a confusion matrix for a binary classification model using the Titanic dataset. A Logistic Regression model was trained to predict passenger survival based on key features (`pclass`, `sex`, `age`, `sibsp`, `parch`, and `fare`).

## Tools and Libraries Used
* **Python**
* **Pandas** (Data loading and preprocessing)
* **Scikit-learn** (Train-test split, Logistic Regression model, and confusion matrix calculation)
* **Matplotlib** (Visualization of the confusion matrix)
* **Jupyter Notebook**

## Model Results and Confusion Matrix Breakdown

The model was evaluated on a test set of 179 passengers, producing the following confusion matrix:

| | Predicted: Did Not Survive (0) | Predicted: Survived (1) |
| :--- | :---: | :---: |
| **Actual: Did Not Survive (0)** | **92** (True Negative) | **13** (False Positive) |
| **Actual: Survived (1)** | **21** (False Negative) | **53** (True Positive) |

### Outcome Categories Explained
* **True Negative (TN) = 92:** The model correctly predicted that 92 passengers did not survive.
* **False Positive (FP) = 13:** The model incorrectly predicted that 13 passengers survived when they actually did not (Type I Error).
* **False Negative (FN) = 21:** The model incorrectly predicted that 21 passengers did not survive when they actually did (Type II Error).
* **True Positive (TP) = 53:** The model correctly predicted that 53 passengers survived.

### Key Performance Observations
* **Total Predictions:** 179 (92 + 13 + 21 + 53)
* **Correct Predictions:** 145 (92 TN + 53 TP)
* **Overall Accuracy:** Approximately 81.01% (145 / 179)
* **Note on Convergence Warning:** During training, Scikit-learn displayed a `ConvergenceWarning` because the `lbfgs` solver reached the default limit of 100 iterations. Setting `max_iter=1000` in `LogisticRegression(max_iter=1000, random_state=42)` or scaling the numerical features will allow the solver to converge cleanly without warnings.

---

## Interview Questions and Answers

### Q1: What is a confusion matrix?
**Answer:** A confusion matrix is a tabular summary used to evaluate the performance of a classification model on test data where the true labels are known. It compares actual ground truth values against model predictions, breaking results down into True Positives, True Negatives, False Positives, and False Negatives.

### Q2: What is a True Positive?
**Answer:** A True Positive occurs when the model predicts a positive outcome and the actual ground truth is also positive. In this Titanic classification task, it represents the 53 passengers whom the model correctly predicted as having survived.

### Q3: What is a False Positive?
**Answer:** A False Positive occurs when the model predicts a positive outcome, but the actual ground truth is negative (also known as a Type I error). In this task, it represents the 13 passengers whom the model predicted would survive, though they actually did not.

### Q4: Why can a confusion matrix be more informative than accuracy?
**Answer:** Accuracy only gives a single percentage of total correct predictions, which can be misleading when dealing with imbalanced datasets. A confusion matrix separates mistakes into False Positives and False Negatives, revealing the exact types of errors the model makes and enabling the calculation of more targeted metrics like precision, recall, and F1-score.
