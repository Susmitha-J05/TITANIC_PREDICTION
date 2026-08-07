"""
data_cleaning.py

Handles loading the raw Titanic dataset and applying the initial cleaning
steps that need to happen *before* the sklearn pipeline (things that are
easier to do with plain pandas): dropping duplicates, engineering a couple
of extra features from the text columns, and getting rid of columns that
are not useful for modelling (IDs, free text, etc).

Everything that has to be "learned" from the training data only (imputation
values, scaling parameters, one-hot categories...) is left to the
scikit-learn Pipeline in train_model.py, so that we never leak information
from the test set into training.
"""

import pandas as pd
import numpy as np


RAW_PATH = "data/titanic.csv"


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a few extra columns from the existing ones.

    - Title: extracted from the passenger's Name (Mr, Mrs, Miss, Master, Rare)
    - FamilySize: SibSp + Parch + 1
    - IsAlone: 1 if travelling alone, 0 otherwise
    - Deck: first letter of the Cabin number (missing -> 'Unknown')
    """
    df = df.copy()

    # Title from name
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
        "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
        "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
    }
    df["Title"] = df["Title"].replace(title_map)
    df.loc[~df["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Title"] = "Rare"

    # Family size / alone flag
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Deck from cabin
    df["Deck"] = df["Cabin"].astype(str).str[0]
    df["Deck"] = df["Deck"].replace("n", "Unknown")  # 'nan' -> 'n' after [0]

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that don't help modelling: identifiers and raw text
    fields we have already extracted information from."""
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows ({before} -> {after})")
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Cap outliers using the IQR rule (1.5 * IQR) instead of removing rows,
    so we don't lose data points, just tame extreme values like Fare."""
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"{col}: capped {n_outliers} outlier values to range [{lower:.2f}, {upper:.2f}]")
    return df


def clean_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    """Full cleaning pipeline that returns a model-ready dataframe.
    Missing-value imputation, encoding, and scaling are intentionally
    NOT done here — they belong in the sklearn Pipeline (see train_model.py)
    so that they are fit only on the training split.
    """
    df = load_raw_data(path)
    df = remove_duplicates(df)
    df = engineer_features(df)
    df = drop_unused_columns(df)
    df = cap_outliers_iqr(df, columns=["Fare", "Age"])
    return df


if __name__ == "__main__":
    data = clean_dataset()
    print(data.head())
    print(data.isnull().sum())
