"""
train_model.py

End-to-end training script for the Titanic survival-prediction project.

Steps:
    1. Load + clean data (src/data_cleaning.py)
    2. Train/test split
    3. Build preprocessing pipeline (imputation, scaling, encoding,
       feature selection) using sklearn's ColumnTransformer + Pipeline
    4. Train & compare 4 classifiers: Logistic Regression, Random Forest,
       XGBoost, and SVM
    5. Evaluate with accuracy, precision, recall, F1 and ROC-AUC
    6. Hyperparameter-tune the best model with GridSearchCV
    7. Save the final pipeline + a metrics report to models/

Run with:  python src/train_model.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_cleaning import clean_dataset

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def get_data():
    df = clean_dataset()
    y = df["Survived"]
    X = df.drop(columns=["Survived"])
    return X, y


NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck"]


# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(k_best=12):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    # Feature selection sits after preprocessing, inside the same pipeline,
    # so it is fit only on the training folds.
    selector = SelectKBest(score_func=f_classif, k=k_best)

    return preprocessor, selector


def make_pipeline(model, k_best=12):
    preprocessor, selector = build_preprocessor(k_best=k_best)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", selector),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# 3. Candidate models
# ---------------------------------------------------------------------------
def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, eval_metric="logloss",
            random_state=RANDOM_STATE, use_label_encoder=False,
        ),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    models = get_candidate_models()
    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        pipe = make_pipeline(model)

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["cv_accuracy_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_accuracy_std"] = round(cv_scores.std(), 4)
        metrics["train_time_sec"] = round(train_time, 3)

        results[name] = metrics
        fitted_pipelines[name] = pipe

        print(f"Test metrics: {metrics}")
        print(confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results).T.sort_values("f1_score", ascending=False)
    print("\n================ Model Comparison ================")
    print(results_df)

    best_model_name = results_df.index[0]
    print(f"\nBest model before tuning: {best_model_name}")

    # -----------------------------------------------------------------
    # 5. Hyperparameter tuning of the best model
    # -----------------------------------------------------------------
    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
        "SVM": {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"],
        },
    }

    base_pipe = make_pipeline(models[best_model_name])
    grid = GridSearchCV(
        base_pipe,
        param_grid=param_grids[best_model_name],
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    print(f"\nRunning GridSearchCV for {best_model_name} ...")
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.4f}")

    tuned_pipe = grid.best_estimator_
    y_pred = tuned_pipe.predict(X_test)
    y_proba = tuned_pipe.predict_proba(X_test)[:, 1]
    tuned_metrics = evaluate(y_test, y_pred, y_proba)
    print(f"Tuned test metrics: {tuned_metrics}")
    print(classification_report(y_test, y_pred))

    # -----------------------------------------------------------------
    # 6. Save artifacts
    # -----------------------------------------------------------------
    joblib.dump(tuned_pipe, MODELS_DIR / "best_pipeline.pkl")

    report = {
        "best_model": best_model_name,
        "model_comparison": results,
        "best_hyperparameters": grid.best_params_,
        "tuned_test_metrics": tuned_metrics,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
    }
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    results_df.to_csv(MODELS_DIR / "model_comparison.csv")

    print(f"\nSaved final pipeline to {MODELS_DIR / 'best_pipeline.pkl'}")
    print(f"Saved training report to {MODELS_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()
