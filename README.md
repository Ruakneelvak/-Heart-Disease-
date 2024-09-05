# Heart Disease Prediction using Machine Learning

### Overview
This project aims to predict the presence of heart disease using various machine learning algorithms. The dataset used is the **Heart Disease dataset**. The models trained include Decision Tree, Support Vector Machine (SVM), Kernel SVM, Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, and Random Forest. Each model's performance is evaluated using confusion matrices and accuracy scores.

### Table of Contents
1. [Dataset](#dataset)
2. [Technologies Used](#technologies-used)
3. [Models Implemented](#models-implemented)
4. [Results](#results)

### Dataset
The dataset used for this project is the **Heart Disease dataset**. It contains multiple features that represent a patient's health attributes, and the target variable represents the presence or absence of heart disease.


### Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Numpy
  - Pandas
  - Matplotlib (for data visualization)
  - Scikit-learn (for machine learning algorithms)


### Models Implemented

- **Decision Tree**:
  - Used entropy as the criterion for splitting.
  - Accuracy: 80.26%

- **Support Vector Machine (SVM)**:
  - Linear kernel.
  - Accuracy: 85.52%

- **Kernel SVM**:
  - Radial basis function (RBF) kernel.
  - Accuracy: 85.52%

- **Logistic Regression**:
  - Accuracy: 82.89%

- **K-Nearest Neighbors (KNN)**:
  - 5 neighbors, Minkowski metric.
  - Accuracy: 81.57%

- **Naive Bayes**:
  - Gaussian Naive Bayes.
  - Accuracy: 82.89%

- **Random Forest**:
  - 10 estimators, entropy criterion.
  - Accuracy: 82.89%

### Results
The models' performance was evaluated using confusion matrices and accuracy scores. Below is a summary of the results:

| Model               | Accuracy  |
|---------------------|-----------|
| Decision Tree        | 80.26%    |
| SVM (Linear Kernel)  | 85.52%    |
| Kernel SVM           | 85.52%    |
| Logistic Regression  | 82.89%    |
| K-Nearest Neighbors  | 81.57%    |
| Naive Bayes          | 82.89%    |
| Random Forest        | 82.89%    |

### Conclusion
The Support Vector Machine (SVM) with both linear and RBF kernels performed the best, achieving an accuracy of **85.52%**. Other models also provided reasonable results and can be tuned further to potentially improve accuracy.
