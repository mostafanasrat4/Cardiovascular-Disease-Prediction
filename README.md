
# 🫀 Cardiovascular Disease Prediction Using Machine Learning

This project applies machine learning techniques to predict 10-year Coronary Heart Disease (CHD) risk using clinical and lifestyle data from the Framingham Heart Study.  
We demonstrate the importance of data preprocessing, model selection, evaluation, and feature analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Authors](#authors)

## 📖 Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally.  
Using machine learning, this project aims to build predictive models that can assist healthcare providers in early diagnosis and risk management.

## 🏥 Dataset
- Source: [Framingham Heart Study](https://www.framinghamheartstudy.org/)
- Instances: **4240 patients**
- Features: **16 attributes** (e.g., age, cholesterol, blood pressure, smoking status)
- Target variable: **`TenYearCHD`** (binary: 1 = High Risk, 0 = Low Risk)

## 🛠 Technologies Used
- **Python 3.9.12**
- **Libraries:**
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn`
  - `matplotlib`

## ⚙️ Installation
To run the project locally:
```bash
git clone https://github.com/your-username/framingham-cvd-prediction.git
cd framingham-cvd-prediction
pip install -r requirements.txt
```

## 📂 Project Structure
```
├── framingham4.ipynb      # Full Jupyter Notebook code
├── report.pdf             # Detailed academic project report
├── README.md              # Documentation
└── requirements.txt       # Python dependencies
```

## 🧹 Data Preprocessing
1. **Handle Missing Values**
```python
from sklearn.impute import SimpleImputer

imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

# Impute numerical features
X[numerical_features] = imputer_num.fit_transform(X[numerical_features])

# Impute categorical features
X[categorical_features] = imputer_cat.fit_transform(X[categorical_features])
```

2. **Outlier Removal**
```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

3. **Feature Scaling**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
```

4. **Balancing Dataset (SMOTE)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

## 🤖 Model Training
**Models Implemented:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

**Example: Random Forest Training**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
```

## 📈 Evaluation Metrics
We evaluated models using:
- 5-Fold Cross Validation
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score

**Example: Generating Evaluation Report**
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## 🏆 Results
| Model               | Cross-Validation Accuracy | Test Accuracy | Best Metric |
|---------------------|----------------------------|---------------|-------------|
| Logistic Regression | 68%                        | 68%           | Simplicity |
| K-Nearest Neighbors | 81%                        | 81%           | Local Sensitivity |
| **Random Forest**   | **87%**                    | **89%**       | Overall Performance |

- **Random Forest** was the best-performing model, achieving:
  - 89% Testing Accuracy
  - Balanced Precision and Recall
  - Robustness to feature interactions and noise

## 🚀 Future Work
- Feature selection techniques (LASSO, PCA)
- Explore Deep Learning models (e.g., Neural Networks with Attention)
- Data Augmentation using GANs
- Hyperparameter Tuning (Grid Search, Bayesian Optimization)

## 👨‍💻 Authors
- [Mostafa Nasrat](https://www.linkedin.com/in/mostafanasrat4/)
- Omar Magdy Mostafa
- AbdELRahman Yasser
- Sherif Essam

> 📚 Developed as part of the Bioinformatics course at Faculty of Engineering, Ain Shams University.
