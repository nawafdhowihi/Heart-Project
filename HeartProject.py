from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "heart_cleveland_upload.csv"
CSS_FILE = BASE_DIR / "style.css"

if CSS_FILE.exists():
    st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
PRIMARY = "#B4233C"
SECONDARY = "#246BFD"
SUCCESS = "#14804A"
TEXT = "#172033"
MUTED = "#667085"
BACKGROUND = "#F6F8FC"

CP_LABELS = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic",
}

SEX_LABELS = {0: "Female", 1: "Male"}
CONDITION_LABELS = {0: "No Heart Disease", 1: "Heart Disease"}

MODEL_FEATURES = ["age", "sex", "cp", "thalach", "exang", "oldpeak", "thal"]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    X = df[MODEL_FEATURES]
    y = df["condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    numeric_features = ["age", "thalach", "oldpeak"]
    categorical_numeric_features = ["sex", "cp", "exang", "thal"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_numeric_features),
        ]
    )

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    tree = DecisionTreeClassifier(random_state=42)

    logistic.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    logistic_pred = logistic.predict(X_test)
    tree_pred = tree.predict(X_test)

    results = {
        "Logistic Regression": {
            "model": logistic,
            "pred": logistic_pred,
            "accuracy": accuracy_score(y_test, logistic_pred),
            "precision": precision_score(y_test, logistic_pred, zero_division=0),
            "recall": recall_score(y_test, logistic_pred, zero_division=0),
            "f1": f1_score(y_test, logistic_pred, zero_division=0),
            "cm": confusion_matrix(y_test, logistic_pred),
            "report": classification_report(
                y_test,
                logistic_pred,
                target_names=["No Disease", "Disease"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "Decision Tree": {
            "model": tree,
            "pred": tree_pred,
            "accuracy": accuracy_score(y_test, tree_pred),
            "precision": precision_score(y_test, tree_pred, zero_division=0),
            "recall": recall_score(y_test, tree_pred, zero_division=0),
            "f1": f1_score(y_test, tree_pred, zero_division=0),
            "cm": confusion_matrix(y_test, tree_pred),
            "report": classification_report(
                y_test,
                tree_pred,
                target_names=["No Disease", "Disease"],
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    return results, X_test, y_test


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filter patients")

    age_min = int(df["age"].min())
    age_max = int(df["age"].max())
    age_range = st.sidebar.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )

    gender_options = ["Female", "Male"]
    selected_gender = st.sidebar.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options,
    )

    cp_options = list(CP_LABELS.values())
    selected_cp = st.sidebar.multiselect(
        "Chest pain type",
        options=cp_options,
        default=cp_options,
    )

    condition_options = list(CONDITION_LABELS.values())
    selected_condition = st.sidebar.multiselect(
        "Condition",
        options=condition_options,
        default=condition_options,
    )

    gender_values = [key for key, value in SEX_LABELS.items() if value in selected_gender]
    cp_values = [key for key, value in CP_LABELS.items() if value in selected_cp]
    condition_values = [
        key for key, value in CONDITION_LABELS.items() if value in selected_condition
    ]

    filtered = df[
        df["age"].between(age_range[0], age_range[1])
        & df["sex"].isin(gender_values)
        & df["cp"].isin(cp_values)
        & df["condition"].isin(condition_values)
    ].copy()

    return filtered


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(icon: str, title: str, text: str):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-icon">{icon}</div>
            <div>
                <h4>{title}</h4>
                <p>{text}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_confusion_matrix(cm: np.ndarray, title: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: No Disease", "Predicted: Disease"],
            y=["Actual: No Disease", "Actual: Disease"],
            text=cm,
            texttemplate="%{text}",
            colorscale=[[0, "#FFF1F3"], [1, PRIMARY]],
            showscale=False,
            hovertemplate="Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
if not DATA_FILE.exists():
    st.error(
        "The dataset file was not found. Place `heart_cleveland_upload.csv` "
        "in the same folder as `app.py`."
    )
    st.stop()

df = load_data(DATA_FILE)
model_results, X_test, y_test = train_models(df)

df_view = df.copy()
df_view["Gender"] = df_view["sex"].map(SEX_LABELS)
df_view["Chest Pain"] = df_view["cp"].map(CP_LABELS)
df_view["Diagnosis"] = df_view["condition"].map(CONDITION_LABELS)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">♥</div>
            <div>
                <div class="sidebar-title">CardioInsight</div>
                <div class="sidebar-subtitle">Heart analytics dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-info">
            Explore patient data, compare clinical indicators, and review
            machine-learning results.
        </div>
        """,
        unsafe_allow_html=True,
    )

filtered_df = apply_filters(df_view)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Dataset")
    st.caption("Heart Disease Cleveland UCI")
    st.caption(f"{len(df):,} patient records • {df.shape[1]} columns")
    st.markdown(
        """
        <div class="medical-note">
            <strong>Educational use only</strong><br>
            Predictions shown here are not a medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">HEALTH DATA • MACHINE LEARNING</div>
        <h1>Heart Disease Analytics Dashboard</h1>
        <p>
            An interactive view of patient characteristics, clinical patterns,
            and classification model performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters.")
    st.stop()


# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------
total_patients = len(filtered_df)
disease_cases = int((filtered_df["condition"] == 1).sum())
healthy_cases = int((filtered_df["condition"] == 0).sum())
disease_rate = disease_cases / total_patients * 100
avg_age = filtered_df["age"].mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Filtered Patients", f"{total_patients:,}", "Current dashboard view")
with k2:
    metric_card("Heart Disease Cases", f"{disease_cases:,}", f"{disease_rate:.1f}% of filtered records")
with k3:
    metric_card("No Disease", f"{healthy_cases:,}", "Patients without the condition")
with k4:
    metric_card("Average Age", f"{avg_age:.1f}", "Years")


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
overview_tab, eda_tab, model_tab, prediction_tab, insights_tab = st.tabs(
    ["Overview", "Exploratory Analysis", "Models", "Risk Prediction", "Insights"]
)


# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------
with overview_tab:
    section_title(
        "Dataset Overview",
        "A quick summary of the selected patient records and overall data quality.",
    )

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("#### Patient data preview")
        preview_columns = [
            "age",
            "Gender",
            "Chest Pain",
            "trestbps",
            "chol",
            "thalach",
            "Diagnosis",
        ]
        st.dataframe(
            filtered_df[preview_columns].head(12),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("#### Data quality")
        q1, q2 = st.columns(2)
        with q1:
            metric_card("Missing Values", f"{int(df.isna().sum().sum())}", "Across all columns")
        with q2:
            metric_card("Duplicate Rows", f"{int(df.duplicated().sum())}", "Exact duplicated records")

        st.markdown("#### Dataset description")
        st.write(
            "The Cleveland heart disease dataset contains demographic and clinical "
            "measurements used to study the presence of heart disease. The target "
            "variable is binary: 0 indicates no heart disease and 1 indicates heart disease."
        )

    st.markdown("#### Summary statistics")
    st.dataframe(
        filtered_df[
            ["age", "trestbps", "chol", "thalach", "oldpeak"]
        ].describe().T.round(2),
        use_container_width=True,
    )


# ---------------------------------------------------------
# EDA
# ---------------------------------------------------------
with eda_tab:
    section_title(
        "Exploratory Data Analysis",
        "Interactive charts reveal how patient features differ by heart-disease status.",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        condition_counts = (
            filtered_df["Diagnosis"].value_counts().rename_axis("Diagnosis").reset_index(name="Count")
        )
        fig_condition = px.bar(
            condition_counts,
            x="Diagnosis",
            y="Count",
            text="Count",
            color="Diagnosis",
            color_discrete_map={
                "No Heart Disease": "#2E8B73",
                "Heart Disease": PRIMARY,
            },
            title="Heart Disease Distribution",
        )
        fig_condition.update_traces(textposition="outside")
        fig_condition.update_layout(showlegend=False, height=410)
        st.plotly_chart(fig_condition, use_container_width=True)

    with chart_col2:
        fig_age = px.histogram(
            filtered_df,
            x="age",
            color="Diagnosis",
            nbins=16,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={
                "No Heart Disease": "#2E8B73",
                "Heart Disease": PRIMARY,
            },
            title="Age Distribution by Diagnosis",
        )
        fig_age.update_layout(height=410, xaxis_title="Age", yaxis_title="Patients")
        st.plotly_chart(fig_age, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig_cp = px.histogram(
            filtered_df,
            x="Chest Pain",
            color="Diagnosis",
            barmode="group",
            color_discrete_map={
                "No Heart Disease": "#2E8B73",
                "Heart Disease": PRIMARY,
            },
            title="Chest Pain Type vs Heart Disease",
        )
        fig_cp.update_layout(height=430, xaxis_title="", yaxis_title="Patients")
        st.plotly_chart(fig_cp, use_container_width=True)

    with chart_col4:
        fig_chol = px.box(
            filtered_df,
            x="Diagnosis",
            y="chol",
            color="Diagnosis",
            points="outliers",
            color_discrete_map={
                "No Heart Disease": "#2E8B73",
                "Heart Disease": PRIMARY,
            },
            title="Cholesterol Level vs Heart Disease",
        )
        fig_chol.update_layout(height=430, showlegend=False, yaxis_title="Cholesterol (mg/dL)")
        st.plotly_chart(fig_chol, use_container_width=True)

    chart_col5 = st.columns(1)[0]

    with chart_col5:
        gender_counts = (
            filtered_df.groupby(["Gender", "Diagnosis"])
            .size()
            .reset_index(name="Count")
        )
        fig_gender = px.bar(
            gender_counts,
            x="Gender",
            y="Count",
            color="Diagnosis",
            barmode="group",
            text="Count",
            color_discrete_map={
                "No Heart Disease": "#2E8B73",
                "Heart Disease": PRIMARY,
            },
            title="Gender vs Heart Disease",
        )
        fig_gender.update_layout(height=430)
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("#### Correlation with Heart Disease")
    corr = (
        df.corr(numeric_only=True)["condition"]
        .drop("condition")
        .sort_values()
        .reset_index()
    )
    corr.columns = ["Feature", "Correlation"]
    fig_corr = px.bar(
        corr,
        x="Correlation",
        y="Feature",
        orientation="h",
        color="Correlation",
        color_continuous_scale=["#246BFD", "#F2F4F7", PRIMARY],
        range_color=[-1, 1],
    )
    fig_corr.update_layout(height=520, coloraxis_showscale=False)
    st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
with model_tab:
    section_title(
        "Machine Learning Performance",
        "Logistic Regression and Decision Tree are trained using the same selected features.",
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": values["accuracy"],
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1 Score": values["f1"],
            }
            for name, values in model_results.items()
        ]
    )

    m1, m2 = st.columns(2)
    for col, model_name in zip([m1, m2], ["Logistic Regression", "Decision Tree"]):
        with col:
            result = model_results[model_name]
            st.markdown(f"#### {model_name}")
            a, b = st.columns(2)
            with a:
                metric_card("Accuracy", f"{result['accuracy'] * 100:.2f}%", "Correct predictions")
            with b:
                metric_card("F1 Score", f"{result['f1'] * 100:.2f}%", "Balance of precision and recall")
            st.plotly_chart(
                plot_confusion_matrix(result["cm"], f"{model_name} Confusion Matrix"),
                use_container_width=True,
            )

    st.markdown("#### Model comparison")
    comparison_long = comparison.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score",
    )
    fig_models = px.bar(
        comparison_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=comparison_long["Score"].map(lambda value: f"{value:.2f}"),
        color_discrete_map={
            "Logistic Regression": SECONDARY,
            "Decision Tree": PRIMARY,
        },
    )
    fig_models.update_layout(yaxis_tickformat=".0%", height=450)
    st.plotly_chart(fig_models, use_container_width=True)

    best_model = comparison.sort_values("Accuracy", ascending=False).iloc[0]
    st.success(
        f"Best model in this split: **{best_model['Model']}** "
        f"with **{best_model['Accuracy'] * 100:.2f}% accuracy**."
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
with prediction_tab:
    section_title(
        "Patient Risk Prediction",
        "Enter the selected clinical features to generate an educational model prediction.",
    )

    st.markdown(
        """
        <div class="prediction-note">
            This prediction is produced by a student machine-learning model and must not
            be used as medical advice or a clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=55)
            sex_label = st.selectbox("Gender", ["Female", "Male"])
            cp_label = st.selectbox("Chest pain type", list(CP_LABELS.values()))

        with c2:
            thalach = st.number_input(
                "Maximum heart rate achieved",
                min_value=60,
                max_value=220,
                value=150,
            )
            exang_label = st.selectbox("Exercise-induced angina", ["No", "Yes"])
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )

        with c3:
            thal = st.selectbox(
                "Thal test value",
                options=sorted(df["thal"].unique().tolist()),
            )
            prediction_model_name = st.selectbox(
                "Prediction model",
                options=["Logistic Regression", "Decision Tree"],
            )

        submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submitted:
        sex = 0 if sex_label == "Female" else 1
        cp = next(key for key, value in CP_LABELS.items() if value == cp_label)
        exang = 0 if exang_label == "No" else 1

        patient = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "thalach": thalach,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "thal": thal,
                }
            ]
        )

        chosen_model = model_results[prediction_model_name]["model"]
        prediction = int(chosen_model.predict(patient)[0])

        if hasattr(chosen_model, "predict_proba"):
            probability = float(chosen_model.predict_proba(patient)[0][1])
        else:
            probability = None

        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-result risk">
                    <div class="prediction-result-title">Higher predicted risk</div>
                    <div class="prediction-result-value">Heart Disease</div>
                    <div class="prediction-result-note">
                        Model: {prediction_model_name}
                        {"• Estimated probability: " + f"{probability * 100:.1f}%" if probability is not None else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-result safe">
                    <div class="prediction-result-title">Lower predicted risk</div>
                    <div class="prediction-result-value">No Heart Disease</div>
                    <div class="prediction-result-note">
                        Model: {prediction_model_name}
                        {"• Estimated disease probability: " + f"{probability * 100:.1f}%" if probability is not None else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# Insights
# ---------------------------------------------------------
with insights_tab:
    section_title(
        "Key Insights",
        "The main findings from the exploratory analysis and model comparison.",
    )

    i1, i2 = st.columns(2)
    with i1:
        insight_card(
            "⚖️",
            "Balanced target",
            "The dataset contains similar proportions of patients with and without heart disease, supporting a fair classification comparison.",
        )
        insight_card(
            "🫀",
            "Chest pain matters",
            "Chest pain type shows a noticeable relationship with the target and contributes useful information to the classification model.",
        )
        insight_card(
            "📉",
            "Maximum heart rate",
            "Maximum heart rate has a negative correlation with heart disease in this dataset.",
        )

    with i2:
        insight_card(
            "🎂",
            "Age pattern",
            "Heart-disease cases are more common among middle-aged and older patients in the dataset.",
        )
        insight_card(
            "🧪",
            "Cholesterol overlap",
            "Cholesterol values overlap between both groups, suggesting that cholesterol alone is not enough for reliable prediction.",
        )
        insight_card(
            "🤖",
            "Model comparison",
            "Logistic Regression provides a stronger baseline than the untuned Decision Tree for the selected train-test split.",
        )

    st.markdown("#### Limitations")
    st.write(
        "The dataset contains only 297 records and represents a limited patient population. "
        "Model performance may change with a larger dataset, different features, or another "
        "train-test split."
    )

    st.markdown("#### Recommended future work")
    st.write(
        "Future versions could include cross-validation, hyperparameter tuning, additional "
        "classification models, and external validation using a larger medical dataset."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <strong>CardioInsight</strong> · Developed by Nawaf Aldhowaihi<br>
        Data analysis and machine learning project · Educational use only
    </div>
    """,
    unsafe_allow_html=True,
)
