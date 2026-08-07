# Titanic Survival Prediction – End-to-End ML Project

Predicting whether a Titanic passenger survived, using a full machine learning
workflow: data cleaning, feature engineering, a scikit-learn preprocessing +
modelling pipeline, comparison of 4 classifiers, hyperparameter tuning, and a
Streamlit app for live predictions.

## Dataset

Titanic passenger dataset (891 rows, 12 columns) — a classic real-world
dataset with:

- **Missing values**: `Age` (177 missing), `Cabin` (687 missing), `Embarked` (2 missing)
- **Categorical features**: `Sex`, `Embarked`, `Pclass`, `Ticket`, `Cabin`
- **Numerical features**: `Age`, `Fare`, `SibSp`, `Parch`
- **Target**: `Survived` (0 = did not survive, 1 = survived), 549 vs 342 — moderately imbalanced

Source file used: `data/titanic.csv`.

## Project Structure

```
titanic_project/
├── data/
│   └── titanic.csv                  # Raw dataset
├── notebooks/
│   └── titanic_end_to_end.ipynb     # Full EDA + training walkthrough (executed, with outputs)
├── src/
│   ├── data_cleaning.py             # Loading, duplicate removal, feature engineering, outlier capping
│   └── train_model.py               # Preprocessing pipeline, model training, tuning, saving artifacts
├── models/
│   ├── best_pipeline.pkl            # Final tuned sklearn Pipeline (preprocessing + model)
│   ├── training_report.json         # Metrics for all models + best hyperparameters
│   └── model_comparison.csv         # Model comparison table
├── app/
│   └── streamlit_app.py             # Streamlit UI for live predictions + performance dashboard
├── requirements.txt
└── README.md
```

## Workflow

### 1. Data Cleaning (`src/data_cleaning.py`)

- **Duplicates**: checked and removed with `drop_duplicates()` (none found in this dataset,
  but the step is always run so the pipeline is robust to other data pulls)
- **Feature engineering**:
  - `Title` – extracted from the `Name` field (Mr / Mrs / Miss / Master / Rare)
  - `FamilySize` – `SibSp + Parch + 1`
  - `IsAlone` – 1 if travelling alone, else 0
  - `Deck` – first letter of the `Cabin` code (`Unknown` if missing)
- **Dropped columns**: `PassengerId`, `Name`, `Ticket`, `Cabin` (IDs / free text
  already captured by the engineered features above)
- **Outlier treatment**: `Age` and `Fare` capped using the IQR rule
  (values outside `Q1 − 1.5×IQR` / `Q3 + 1.5×IQR` are clipped, not dropped, so
  no rows are lost)

Missing-value imputation, one-hot encoding and feature scaling are **not**
done in this step. They are handled inside the scikit-learn `Pipeline`
(next section) so that they are only ever fit on the training data — this
avoids any leakage from the test set into preprocessing.

### 2. Preprocessing Pipeline (`src/train_model.py`)

A single `sklearn.pipeline.Pipeline` wraps everything:

```
ColumnTransformer
 ├── numeric features   -> SimpleImputer(median) -> StandardScaler
 └── categorical features -> SimpleImputer(most_frequent) -> OneHotEncoder
        |
        v
SelectKBest(f_classif, k=12)   # feature selection
        |
        v
Classifier (LogReg / RandomForest / XGBoost / SVM)
```

- **Numeric features**: `Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`, `IsAlone`
- **Categorical features**: `Pclass`, `Sex`, `Embarked`, `Title`, `Deck`

Because everything lives inside one `Pipeline` object, `pipeline.fit(X_train, y_train)`
and `pipeline.predict(X_new)` handle the entire preprocessing + inference chain —
this is the exact same object saved to `models/best_pipeline.pkl` and used by
the Streamlit app, so there is no risk of train/inference preprocessing mismatch.

### 3. Model Comparison

Four classifiers were trained with identical preprocessing and evaluated with
5-fold cross-validation (train set) and a held-out 20% test set:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression   | 0.7821 | 0.7206 | 0.7101 | 0.7153 | 0.8418 |
| Random Forest          | 0.7933 | 0.7424 | 0.7101 | 0.7259 | 0.8182 |
| XGBoost                | 0.7877 | 0.7313 | 0.7101 | 0.7206 | 0.8265 |
| SVM                     | 0.7933 | 0.8077 | 0.6087 | 0.6942 | 0.8588 |

(Full table with CV scores and training time is in `models/model_comparison.csv`.)

**Best model (by F1-score): Random Forest**

### 4. Hyperparameter Tuning

`GridSearchCV` (5-fold, scoring = F1) was run over Random Forest's
`n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

Best parameters found:
```
n_estimators: 500
max_depth: None
min_samples_split: 5
min_samples_leaf: 1
```

**Test-set performance after tuning:**

| Metric    | Before tuning | After tuning |
|-----------|:-------------:|:------------:|
| Accuracy  | 0.7933        | 0.8101       |
| Precision | 0.7424        | 0.7612       |
| Recall    | 0.7101        | 0.7391       |
| F1 Score  | 0.7259        | 0.7500       |
| ROC-AUC   | 0.8182        | 0.8252       |

Tuning improved every metric, most notably accuracy (+1.7pp) and F1 (+2.4pp).

### 5. Streamlit App (`app/streamlit_app.py`)

Two tabs:
- **Predict**: form for passenger class, sex, age, family size, fare, port of
  embarkation, title, and deck → live survival prediction with probability
- **Model Performance**: comparison table for all 4 models, tuned metrics,
  best hyperparameters, and the feature list used

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (regenerates everything in `models/`)
```bash
python src/train_model.py
```

### 3. Explore the notebook
```bash
jupyter notebook notebooks/titanic_end_to_end.ipynb
```

### 4. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```
Then open the printed local URL (usually `http://localhost:8501`) in your browser.

## Key Design Decisions

- **Outlier capping instead of row removal** for `Age`/`Fare`, to avoid losing
  already-limited training data (only 891 rows total).
- **Feature engineering from `Name`/`Cabin`** (`Title`, `Deck`) instead of
  dropping them outright, since they carry real predictive signal (e.g. "Master"
  title strongly correlates with young boys, higher decks correlate with
  higher class / survival).
- **Imputation, encoding and scaling inside the pipeline**, never before the
  train/test split, to prevent data leakage.
- **SelectKBest feature selection** inside the pipeline keeps only the most
  informative encoded features, reducing noise from the one-hot expansion.
- **F1-score used for model selection/tuning** rather than plain accuracy,
  since the target classes are imbalanced (61% / 39%).

## Results Summary

The final tuned Random Forest pipeline achieves **81% accuracy** and **0.75 F1**
on the held-out test set, with an ROC-AUC of 0.825 — a solid result for this
well-known benchmark dataset given the relatively small size (891 rows) and
inherent noise in survival outcomes.
