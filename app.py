
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(
    page_title="Telecom Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telecom Customer Churn Dashboard")
st.caption("Based on the analysis in Telecom_churn.ipynb")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Match the notebook cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().copy()
    return df

@st.cache_resource
def train_models(df):
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5,
            min_samples_leaf=3, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        )
    }

    fitted = {}
    rows = []

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        fitted[name] = pipe
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })

    results = pd.DataFrame(rows)
    best_name = results.loc[results["F1 Score"].idxmax(), "Model"]

    return fitted, results, best_name, X_test, y_test

# Sidebar
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload the telecom CSV",
    type=["csv"],
    help="Use WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df = load_data(uploaded_file)

# Sidebar filters
contract_options = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
selected_contract = st.sidebar.selectbox("Contract", contract_options)

gender_options = ["All"] + sorted(df["gender"].dropna().unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", gender_options)

if selected_contract != "All":
    view_df = df[df["Contract"] == selected_contract].copy()
else:
    view_df = df.copy()

if selected_gender != "All":
    view_df = view_df[view_df["gender"] == selected_gender].copy()

# KPI row
total_customers = len(view_df)
churned = int((view_df["Churn"] == "Yes").sum())
churn_rate = (churned / total_customers * 100) if total_customers else 0
avg_tenure = view_df["tenure"].mean() if total_customers else 0
avg_monthly = view_df["MonthlyCharges"].mean() if total_customers else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{total_customers:,}")
c2.metric("Churned Customers", f"{churned:,}")
c3.metric("Churn Rate", f"{churn_rate:.1f}%")
c4.metric("Avg Monthly Charges", f"${avg_monthly:,.2f}")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", "🔍 Churn Analysis", "🤖 Model Performance", "👤 Customer Prediction"
])

with tab1:
    st.subheader("Customer Churn Distribution")

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = view_df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Customers"]
        fig = px.bar(
            churn_counts, x="Churn", y="Customers",
            color="Churn", text="Customers",
            title="Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        contract_churn = (
            view_df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Customer Churn by Contract Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Data Preview")
    st.dataframe(view_df.head(100), use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            view_df,
            x="Churn",
            y="tenure",
            color="Churn",
            title="Customer Tenure vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            view_df,
            x="MonthlyCharges",
            color="Churn",
            nbins=30,
            marginal="box",
            title="Monthly Charges vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Rate by Contract")
    contract_rate = (
        view_df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )
    fig = px.bar(
        contract_rate,
        x="Contract",
        y="Churn Rate",
        text_auto=".1f",
        title="Churn Rate by Contract Type"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Model Comparison")

    with st.spinner("Training models..."):
        models, results, best_model_name, X_test, y_test = train_models(df)

    display_results = results.copy()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    display_results[metric_cols] = display_results[metric_cols].round(4)

    st.dataframe(display_results, use_container_width=True)

    fig = px.bar(
        results.melt(
            id_vars="Model",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score"
        ),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 1],
        title="Model Performance Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Best model by F1 Score: {best_model_name}")

    # Confusion matrix for best model
    best_model = models[best_model_name]
    pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"]
    )

    st.subheader(f"Confusion Matrix — {best_model_name}")
    st.dataframe(cm_df, use_container_width=True)

with tab4:
    st.subheader("Predict Churn for an Individual Customer")

    with st.spinner("Preparing prediction model..."):
        models, results, best_model_name, _, _ = train_models(df)

    model = models[best_model_name]

    # Build one-row input using the original feature names
    input_data = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["gender"] = st.selectbox("Gender", sorted(df["gender"].unique()))
        input_data["SeniorCitizen"] = st.selectbox("Senior Citizen", sorted(df["SeniorCitizen"].unique()))
        input_data["Partner"] = st.selectbox("Partner", sorted(df["Partner"].unique()))
        input_data["Dependents"] = st.selectbox("Dependents", sorted(df["Dependents"].unique()))
        input_data["tenure"] = st.slider("Tenure (months)", 0, int(df["tenure"].max()), int(df["tenure"].median()))

    with col2:
        input_data["PhoneService"] = st.selectbox("Phone Service", sorted(df["PhoneService"].unique()))
        input_data["MultipleLines"] = st.selectbox("Multiple Lines", sorted(df["MultipleLines"].unique()))
        input_data["InternetService"] = st.selectbox("Internet Service", sorted(df["InternetService"].unique()))
        input_data["OnlineSecurity"] = st.selectbox("Online Security", sorted(df["OnlineSecurity"].unique()))
        input_data["OnlineBackup"] = st.selectbox("Online Backup", sorted(df["OnlineBackup"].unique()))
        input_data["DeviceProtection"] = st.selectbox("Device Protection", sorted(df["DeviceProtection"].unique()))

    with col3:
        input_data["TechSupport"] = st.selectbox("Tech Support", sorted(df["TechSupport"].unique()))
        input_data["StreamingTV"] = st.selectbox("Streaming TV", sorted(df["StreamingTV"].unique()))
        input_data["StreamingMovies"] = st.selectbox("Streaming Movies", sorted(df["StreamingMovies"].unique()))
        input_data["Contract"] = st.selectbox("Contract", sorted(df["Contract"].unique()))
        input_data["PaperlessBilling"] = st.selectbox("Paperless Billing", sorted(df["PaperlessBilling"].unique()))
        input_data["PaymentMethod"] = st.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()))

    col4, col5 = st.columns(2)
    with col4:
        input_data["MonthlyCharges"] = st.number_input(
            "Monthly Charges", min_value=0.0,
            value=float(df["MonthlyCharges"].median())
        )
    with col5:
        input_data["TotalCharges"] = st.number_input(
            "Total Charges", min_value=0.0,
            value=float(df["TotalCharges"].median())
        )

    customer = pd.DataFrame([input_data])

    if st.button("Predict Churn", type="primary"):
        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0, 1])

        if prediction == 1:
            st.error(f"⚠️ Predicted Churn — probability: {probability:.1%}")
        else:
            st.success(f"✅ Predicted No Churn — churn probability: {probability:.1%}")

        st.progress(probability, text=f"Churn probability: {probability:.1%}")

st.sidebar.caption("Dashboard follows the data-cleaning and model-comparison approach in the supplied notebook.")
