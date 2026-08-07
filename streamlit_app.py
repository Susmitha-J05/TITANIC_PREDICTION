"""
Streamlit app for the Titanic survival prediction project.

Loads the trained pipeline (models/best_pipeline.pkl) and lets the user enter
passenger details to get a live survival prediction, along with a summary of
how the model performed during training/evaluation.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.pkl"
REPORT_PATH = BASE_DIR / "models" / "training_report.json"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_report():
    with open(REPORT_PATH) as f:
        return json.load(f)


def build_title(name_title_choice):
    return name_title_choice


def engineer_input_row(pclass, sex, age, sibsp, parch, fare, embarked, title, deck):
    """Turn raw form inputs into the same engineered feature row the model
    was trained on (see src/data_cleaning.py -> engineer_features)."""
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    row = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Title": title,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Deck": deck,
    }
    return pd.DataFrame([row])


def main():
    st.title("🚢 Titanic Survival Predictor")
    st.write(
        "This app uses a scikit-learn pipeline (imputation + scaling + "
        "encoding + feature selection + classifier) trained on the Titanic "
        "dataset to estimate a passenger's chance of survival."
    )

    if not MODEL_PATH.exists():
        st.error(
            "No trained model found. Please run `python src/train_model.py` "
            "first to generate models/best_pipeline.pkl."
        )
        return

    pipeline = load_pipeline()
    report = load_report()

    tab_predict, tab_performance = st.tabs(["🔮 Predict", "📊 Model Performance"])

    # ------------------------------------------------------------------
    # Prediction tab
    # ------------------------------------------------------------------
    with tab_predict:
        st.subheader("Enter passenger details")

        col1, col2, col3 = st.columns(3)

        with col1:
            pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2,
                                   help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
            sex = st.selectbox("Sex", options=["male", "female"])
            age = st.slider("Age", min_value=0, max_value=80, value=28)

        with col2:
            sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=8, value=0)
            parch = st.number_input("Parents / Children aboard", min_value=0, max_value=6, value=0)
            fare = st.slider("Fare paid (£)", min_value=0.0, max_value=250.0, value=32.0, step=0.5)

        with col3:
            embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"],
                                     format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
            title = st.selectbox("Title", options=["Mr", "Mrs", "Miss", "Master", "Rare"])
            deck = st.selectbox("Deck", options=["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"])

        st.markdown("---")

        if st.button("Predict Survival", type="primary"):
            input_df = engineer_input_row(pclass, sex, age, sibsp, parch, fare, embarked, title, deck)

            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0]

            survive_prob = probability[1]
            die_prob = probability[0]

            result_col, chart_col = st.columns([1, 1])

            with result_col:
                if prediction == 1:
                    st.success(f"### Predicted: SURVIVED ✅\nEstimated probability: **{survive_prob:.1%}**")
                else:
                    st.error(f"### Predicted: DID NOT SURVIVE ❌\nEstimated probability of not surviving: **{die_prob:.1%}**")

                st.write("Input summary:")
                st.dataframe(input_df, use_container_width=True)

            with chart_col:
                st.write("Survival probability")
                prob_df = pd.DataFrame({
                    "Outcome": ["Did not survive", "Survived"],
                    "Probability": [die_prob, survive_prob],
                })
                st.bar_chart(prob_df.set_index("Outcome"))

    # ------------------------------------------------------------------
    # Model performance tab
    # ------------------------------------------------------------------
    with tab_performance:
        st.subheader("Model comparison (test set)")
        comp_df = pd.DataFrame(report["model_comparison"]).T
        st.dataframe(comp_df, use_container_width=True)

        st.subheader(f"Best model: {report['best_model']} (after hyperparameter tuning)")
        col1, col2, col3, col4, col5 = st.columns(5)
        tuned = report["tuned_test_metrics"]
        col1.metric("Accuracy", f"{tuned['accuracy']:.2%}")
        col2.metric("Precision", f"{tuned['precision']:.2%}")
        col3.metric("Recall", f"{tuned['recall']:.2%}")
        col4.metric("F1 Score", f"{tuned['f1_score']:.2%}")
        col5.metric("ROC-AUC", f"{tuned['roc_auc']:.2%}")

        st.subheader("Best hyperparameters found (GridSearchCV)")
        st.json(report["best_hyperparameters"])

        st.subheader("Features used")
        f1, f2 = st.columns(2)
        with f1:
            st.write("**Numeric features**")
            st.write(report["features_numeric"])
        with f2:
            st.write("**Categorical features**")
            st.write(report["features_categorical"])

    st.markdown("---")
    st.caption(
        "Dataset: Titanic passenger data. Model pipeline includes imputation, "
        "outlier capping, one-hot encoding, scaling, feature selection (SelectKBest) "
        "and a tuned classifier, all wrapped in a single scikit-learn Pipeline."
    )


if __name__ == "__main__":
    main()
