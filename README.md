# 🚢 Titanic Survival Prediction using Machine Learning

This project presents an end-to-end machine learning solution for predicting
whether a passenger survived the Titanic disaster. It covers the complete ML
workflow including data preprocessing, feature engineering, model training,
hyperparameter tuning, evaluation, and deployment through a Streamlit web
application.

---

# 📁 Project Structure

```text
titanic_project/
│
├── data/
│   └── titanic.csv
│
├── notebooks/
│   └── titanic_end_to_end.ipynb
│
├── src/
│   ├── data_cleaning.py
│   └── train_model.py
│
├── models/
│   ├── best_pipeline.pkl
│   ├── training_report.json
│   └── model_comparison.csv
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Overview

The project uses the classic **Titanic** passenger dataset consisting of
**891 records** and **12 original features**.

### Target Variable

- **Survived**
  - 0 → Passenger did not survive
  - 1 → Passenger survived

### Important Features

**Numerical Features**

- Age
- Fare
- SibSp
- Parch

**Categorical Features**

- Passenger Class (Pclass)
- Sex
- Embarked
- Ticket
- Cabin

### Missing Values

The dataset contains several missing values that are handled during
preprocessing.

| Feature | Missing Values |
|----------|---------------:|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

---

# ⚙️ Project Workflow

## 1. Data Cleaning

The data preparation stage performs the following operations:

- Removes duplicate records when present
- Creates additional informative features
- Handles extreme values using IQR-based outlier capping
- Removes unnecessary identifier columns

### Feature Engineering

The following new features are generated:

- **Title** extracted from passenger names
- **FamilySize** calculated from family-related columns
- **IsAlone** indicating whether a passenger travelled alone
- **Deck** extracted from the cabin information

The following columns are removed after feature extraction:

- PassengerId
- Name
- Ticket
- Cabin

Outliers in **Age** and **Fare** are clipped using the Interquartile Range
(IQR) method instead of removing observations.

---

## 2. Data Preprocessing Pipeline

All preprocessing operations are combined into a single
**scikit-learn Pipeline**, ensuring identical processing during training and
prediction.

### Numerical Features

- Median value imputation
- Standard scaling

### Categorical Features

- Most frequent value imputation
- One-Hot Encoding

### Feature Selection

After preprocessing, the pipeline applies:

- **SelectKBest (ANOVA F-test)**

before training the classifier.

The complete preprocessing and prediction workflow is stored inside
`best_pipeline.pkl`, allowing the Streamlit application to use exactly the
same pipeline without additional preprocessing.

---

## 3. Model Development

Four different classification algorithms are trained using the same
preprocessing pipeline.

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)

Training performance is measured using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

# 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.7821 | 0.7206 | 0.7101 | 0.7153 | 0.8418 |
| Random Forest | 0.7933 | 0.7424 | 0.7101 | 0.7259 | 0.8182 |
| XGBoost | 0.7877 | 0.7313 | 0.7101 | 0.7206 | 0.8265 |
| Support Vector Machine | 0.7933 | 0.8077 | 0.6087 | 0.6942 | 0.8588 |

Among the evaluated models, **Random Forest** produced the highest overall
F1-score and was selected as the final model for deployment.

A complete comparison report is available in:

```
models/model_comparison.csv
```

---

# 🔧 Hyperparameter Optimization

The selected Random Forest model is further optimized using
**GridSearchCV** with **5-fold cross-validation**.

### Parameters Tuned

- Number of trees (`n_estimators`)
- Maximum tree depth (`max_depth`)
- Minimum samples required to split
- Minimum samples required in leaf nodes

### Best Parameters

```text
n_estimators = 500
max_depth = None
min_samples_split = 5
min_samples_leaf = 1
```

### Performance Before and After Tuning

| Metric | Initial Model | Tuned Model |
|--------|:-------------:|:-----------:|
| Accuracy | 0.7933 | 0.8101 |
| Precision | 0.7424 | 0.7612 |
| Recall | 0.7101 | 0.7391 |
| F1 Score | 0.7259 | 0.7500 |
| ROC-AUC | 0.8182 | 0.8252 |

Hyperparameter tuning improved every evaluation metric, with the most notable
increase observed in F1-score.

---

# 💻 Streamlit Application

The project includes an interactive web application where users can enter
passenger information to predict survival probability.

### User Inputs

- Passenger Class
- Gender
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Port of Embarkation
- Passenger Title
- Deck

### Application Features

- Predict passenger survival
- Display prediction probability
- View model comparison results
- Display tuned model metrics
- Show selected hyperparameters

---

# 🚀 Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Models

```bash
python src/train_model.py
```

This command:

- cleans the dataset
- engineers new features
- trains all classifiers
- performs hyperparameter tuning
- saves the trained pipeline and evaluation reports

## Open the Notebook

```bash
jupyter notebook notebooks/titanic_end_to_end.ipynb
```

## Launch the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed by Streamlit to interact with the prediction
dashboard.

---

# 📌 Project Highlights

- Duplicate detection and removal
- Feature engineering from passenger information
- IQR-based outlier treatment
- Unified preprocessing with scikit-learn Pipeline
- Automatic missing value imputation
- One-Hot Encoding for categorical variables
- Feature selection using SelectKBest
- Comparison of four machine learning classifiers
- Hyperparameter tuning using GridSearchCV
- Interactive Streamlit deployment

---

# 📊 Final Results

The optimized **Random Forest Pipeline** achieved the strongest overall
performance among the evaluated models.

**Final Test Performance**

- Accuracy: **81.01%**
- Precision: **0.7612**
- Recall: **0.7391**
- F1-Score: **0.7500**
- ROC-AUC: **0.8252**

These results demonstrate that combining feature engineering, pipeline-based
preprocessing, and hyperparameter optimization produces a reliable survival
prediction model for the Titanic dataset.

---

# 🛠️ Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- Streamlit
- joblib
- Jupyter Notebook
