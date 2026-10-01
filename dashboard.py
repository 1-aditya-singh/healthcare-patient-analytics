import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional SciPy
try:
    from scipy.stats import (
        ttest_ind,
        f_oneway,
        chi2_contingency
    )
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Healthcare Patient Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK THEME CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    /* Main content */
    .main {
        background-color: #0b1120;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Headers */
    h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    h2 {
        color: #f1f5f9 !important;
    }

    h3 {
        color: #cbd5e1 !important;
    }

    p {
        color: #cbd5e1;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(
            145deg,
            #111827,
            #172033
        );
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 10px;
        min-height: 125px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }

    .kpi-title {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
    }

    .kpi-subtitle {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    /* Section cards */
    .section-card {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Info boxes */
    .info-box {
        background-color: #111827;
        border-left: 4px solid #64748b;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #263244;
        border-radius: 10px;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #111827;
        border-color: #374151;
    }

    /* Number inputs */
    input {
        background-color: #111827 !important;
        color: #f8fafc !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fafc;
    }

    /* Divider */
    hr {
        border-color: #263244;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background-color: #0b1120 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = Path(
    "data/processed/cleaned_healthcare_patients.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "Dataset not found.\n\n"
        "Expected location:\n"
        "`data/processed/cleaned_healthcare_patients.csv`"
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# ============================================================
# BASIC DATA PREPARATION
# ============================================================

numeric_columns = [
    "age",
    "blood_pressure",
    "heart_rate",
    "bmi",
    "length_of_stay",
    "medication_count",
    "treatment_cost",
    "risk_score"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_currency(value):
    return f"₹{value:,.0f}"


def format_number(value):
    return f"{value:,.0f}"


def make_chart_dark(ax):
    ax.set_facecolor("#111827")
    ax.figure.set_facecolor("#0b1120")

    ax.tick_params(
        colors="#cbd5e1",
        labelcolor="#cbd5e1"
    )

    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#f8fafc")

    for spine in ax.spines.values():
        spine.set_color("#374151")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏥 Healthcare Analytics")

st.sidebar.markdown("---")

st.sidebar.write(
    "Interactive analysis of patient demographics, "
    "diseases, healthcare costs, risk scores, "
    "hospital stays and readmissions."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset")

st.sidebar.write(
    f"Patients: {len(df):,}"
)

st.sidebar.write(
    f"Features: {df.shape[1]}"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

disease_options = sorted(
    df["disease"].dropna().unique()
)

selected_disease = st.sidebar.multiselect(
    "Disease",
    disease_options,
    default=disease_options
)

gender_options = sorted(
    df["gender"].dropna().unique()
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options
)

admission_options = sorted(
    df["admission_type"].dropna().unique()
)

selected_admission = st.sidebar.multiselect(
    "Admission Type",
    admission_options,
    default=admission_options
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    df["disease"].isin(selected_disease)
    & df["gender"].isin(selected_gender)
    & df["admission_type"].isin(selected_admission)
].copy()


# ============================================================
# TITLE
# ============================================================

st.title("🏥 Healthcare Patient Analytics Dashboard")

st.write(
    "Interactive analysis of patient demographics, diseases, "
    "healthcare costs, risk scores, hospital stays and readmissions."
)

st.markdown("---")

st.write(
    f"Showing **{len(filtered_df):,}** patients "
    f"out of **{len(df):,}** total patients."
)


# ============================================================
# KPI SECTION
# ============================================================

st.header("📊 Key Healthcare Metrics")

total_patients = len(filtered_df)

avg_age = filtered_df["age"].mean()

avg_risk = filtered_df["risk_score"].mean()

avg_cost = filtered_df["treatment_cost"].mean()

avg_los = filtered_df["length_of_stay"].mean()

readmission_rate = (
    filtered_df["readmission"].eq("Yes").mean() * 100
)


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    kpi_card(
        "Total Patients",
        f"{total_patients:,}",
        "Filtered patients"
    )

with col2:
    kpi_card(
        "Average Age",
        f"{avg_age:.1f}",
        "Years"
    )

with col3:
    kpi_card(
        "Average Risk",
        f"{avg_risk:.2f}",
        "Risk score"
    )

with col4:
    kpi_card(
        "Average Cost",
        format_currency(avg_cost),
        "Treatment cost"
    )

with col5:
    kpi_card(
        "Readmission Rate",
        f"{readmission_rate:.2f}%",
        "Patients readmitted"
    )


st.markdown("---")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "👥 Demographics",
        "🦠 Disease Analysis",
        "🎯 Risk Analysis",
        "💰 Cost & Hospital Stay",
        "🔄 Readmission",
        "🧪 Statistical Analysis"
    ]
)


# ============================================================
# TAB 1 — DEMOGRAPHICS
# ============================================================

with tabs[0]:

    st.header("👥 Patient Demographics")

    col1, col2 = st.columns(2)

    # Gender
    with col1:

        st.subheader("Gender Distribution")

        gender_counts = (
            filtered_df["gender"]
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.bar(
            gender_counts.index,
            gender_counts.values
        )

        ax.set_title(
            "Patients by Gender",
            fontsize=14
        )

        ax.set_xlabel("Gender")
        ax.set_ylabel("Patient Count")

        make_chart_dark(ax)

        st.pyplot(
            fig,
            clear_figure=True
        )

    # Age
    with col2:

        st.subheader("Age Distribution")

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.hist(
            filtered_df["age"].dropna(),
            bins=20
        )

        ax.set_title(
            "Age Distribution",
            fontsize=14
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Patients")

        make_chart_dark(ax)

        st.pyplot(
            fig,
            clear_figure=True
        )


    st.subheader("Smoking Status")

    smoking_counts = (
        filtered_df["smoking_status"]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        smoking_counts.index,
        smoking_counts.values
    )

    ax.set_title(
        "Smoking Status Distribution"
    )

    ax.set_xlabel("Smoking Status")
    ax.set_ylabel("Patient Count")

    make_chart_dark(ax)

    st.pyplot(
        fig,
        clear_figure=True
    )


    st.subheader("Age Statistics")

    age_stats = pd.DataFrame(
        {
            "Metric": [
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "Standard Deviation"
            ],
            "Value": [
                filtered_df["age"].mean(),
                filtered_df["age"].median(),
                filtered_df["age"].min(),
                filtered_df["age"].max(),
                filtered_df["age"].std()
            ]
        }
    )

    st.dataframe(
        age_stats.round(2),
        hide_index=True
    )


# ============================================================
# TAB 2 — DISEASE ANALYSIS
# ============================================================

with tabs[1]:

    st.header("🦠 Disease Analysis")

    disease_counts = (
        filtered_df["disease"]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        disease_counts.index[::-1],
        disease_counts.values[::-1]
    )

    ax.set_title(
        "Patient Distribution by Disease"
    )

    ax.set_xlabel("Patient Count")
    ax.set_ylabel("Disease")

    make_chart_dark(ax)

    st.pyplot(
        fig,
        clear_figure=True
    )


    st.subheader(
        "Disease-wise Healthcare Metrics"
    )

    disease_summary = (
        filtered_df
        .groupby("disease")
        .agg(
            Patient_Count=("patient_id", "count"),
            Avg_Risk_Score=("risk_score", "mean"),
            Avg_Treatment_Cost=("treatment_cost", "mean"),
            Avg_Length_of_Stay=("length_of_stay", "mean")
        )
        .sort_values(
            "Avg_Risk_Score",
            ascending=False
        )
    )

    st.dataframe(
        disease_summary.round(2)
    )


    st.subheader(
        "Disease vs Risk Score"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.boxplot(
        data=filtered_df,
        x="disease",
        y="risk_score",
        ax=ax
    )

    ax.set_title(
        "Risk Score Distribution by Disease"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    make_chart_dark(ax)

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# TAB 3 — RISK ANALYSIS
# ============================================================

with tabs[2]:

    st.header("🎯 Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Age vs Risk Score"
        )

        correlation = filtered_df[
            ["age", "risk_score"]
        ].corr().iloc[0, 1]

        st.markdown(
            f"""
            <div class="info-box">
                <b>Correlation:</b>
                {correlation:.4f}
            </div>
            """,
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            filtered_df["age"],
            filtered_df["risk_score"],
            alpha=0.35
        )

        ax.set_title(
            "Age vs Risk Score"
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Risk Score")

        make_chart_dark(ax)

        st.pyplot(
            fig,
            clear_figure=True
        )


    with col2:

        st.subheader(
            "Risk Score Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            filtered_df["risk_score"],
            bins=25
        )

        ax.set_title(
            "Risk Score Distribution"
        )

        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Patients")

        make_chart_dark(ax)

        st.pyplot(
            fig,
            clear_figure=True
        )


    st.subheader(
        "Risk Score by Smoking Status"
    )

    smoking_risk = (
        filtered_df
        .groupby("smoking_status")[
            "risk_score"
        ]
        .agg(
            ["count", "mean", "median", "min", "max"]
        )
        .round(2)
    )

    st.dataframe(
        smoking_risk
    )


    st.subheader(
        "Risk Score by Disease"
    )

    risk_disease = (
        filtered_df
        .groupby("disease")[
            "risk_score"
        ]
        .agg(
            ["count", "mean", "median", "min", "max"]
        )
        .sort_values(
            "mean",
            ascending=False
        )
        .round(2)
    )

    st.dataframe(
        risk_disease
    )


# ============================================================
# TAB 4 — COST & HOSPITAL STAY
# ============================================================

with tabs[3]:

    st.header("💰 Cost & Hospital Stay Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Treatment Cost Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            filtered_df["treatment_cost"],
            bins=30
        )

        ax.set_title(
            "Treatment Cost Distribution"
        )

        ax.set_xlabel(
            "Treatment Cost (₹)"
        )

        ax.set_ylabel(
            "Patients"
        )

        make_chart_dark(ax)

        st.pyplot(
            fig,
            clear_figure=True
        )


    with col2:

        st.subheader(
            "Length of Stay Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            filtered_df["length_of_stay"],
            bins=20
        )

        ax.set_title(
            "Hospital Length of Stay"
        )

        ax.set_xlabel(
            "Days"
        )

        ax.set_ylabel(
            "Patients"
        )

        make_chart_dark(ax)

        st.pyplot(
            fig,
            clear_figure=True
        )


    st.subheader(
        "Admission Type Analysis"
    )

    admission_summary = (
        filtered_df
        .groupby("admission_type")
        .agg(
            Patient_Count=("patient_id", "count"),
            Avg_Cost=("treatment_cost", "mean"),
            Avg_Length_of_Stay=(
                "length_of_stay",
                "mean"
            ),
            Avg_Risk=("risk_score", "mean")
        )
        .sort_values(
            "Avg_Cost",
            ascending=False
        )
    )

    st.dataframe(
        admission_summary.round(2)
    )


    st.subheader(
        "Treatment Cost vs Length of Stay"
    )

    cost_stay_corr = filtered_df[
        ["treatment_cost", "length_of_stay"]
    ].corr().iloc[0, 1]

    st.write(
        f"Correlation: **{cost_stay_corr:.4f}**"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        filtered_df["length_of_stay"],
        filtered_df["treatment_cost"],
        alpha=0.35
    )

    ax.set_title(
        "Treatment Cost vs Length of Stay"
    )

    ax.set_xlabel(
        "Length of Stay (Days)"
    )

    ax.set_ylabel(
        "Treatment Cost (₹)"
    )

    make_chart_dark(ax)

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# TAB 5 — READMISSION
# ============================================================

with tabs[4]:

    st.header("🔄 Readmission Analysis")

    readmission_counts = (
        filtered_df["readmission"]
        .value_counts()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Readmission Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.bar(
            readmission_counts.index,
            readmission_counts.values
        )

        ax.set_title(
            "Readmission Status"
        )

        ax.set_xlabel(
            "Readmission"
        )

        ax.set_ylabel(
            "Patients"
        )

        make_chart_dark(ax)

        st.pyplot(
            fig,
            clear_figure=True
        )


    with col2:

        st.subheader(
            "Readmission Summary"
        )

        readmission_summary = (
            filtered_df
            .groupby("readmission")
            .agg(
                Patient_Count=(
                    "patient_id",
                    "count"
                ),
                Avg_Risk=(
                    "risk_score",
                    "mean"
                ),
                Avg_Cost=(
                    "treatment_cost",
                    "mean"
                ),
                Avg_Length_of_Stay=(
                    "length_of_stay",
                    "mean"
                )
            )
            .round(2)
        )

        st.dataframe(
            readmission_summary
        )


    st.subheader(
        "Readmission vs Risk Score"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.boxplot(
        data=filtered_df,
        x="readmission",
        y="risk_score",
        ax=ax
    )

    ax.set_title(
        "Risk Score by Readmission Status"
    )

    make_chart_dark(ax)

    st.pyplot(
        fig,
        clear_figure=True
    )


    st.subheader(
        "Readmission vs Treatment Cost"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.boxplot(
        data=filtered_df,
        x="readmission",
        y="treatment_cost",
        ax=ax
    )

    ax.set_title(
        "Treatment Cost by Readmission Status"
    )

    ax.set_ylabel(
        "Treatment Cost (₹)"
    )

    make_chart_dark(ax)

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# TAB 6 — STATISTICAL ANALYSIS
# ============================================================

with tabs[5]:

    st.header("🧪 Statistical Analysis")

    if not SCIPY_AVAILABLE:

        st.warning(
            "SciPy is not installed. "
            "Install it using:\n\n"
            "`pip install scipy`\n\n"
            "Then restart Streamlit."
        )

    else:

        st.subheader(
            "1. Readmission vs Risk Score"
        )

        readmitted_risk = filtered_df[
            filtered_df["readmission"] == "Yes"
        ]["risk_score"].dropna()

        not_readmitted_risk = filtered_df[
            filtered_df["readmission"] == "No"
        ]["risk_score"].dropna()

        if len(readmitted_risk) > 1 and len(not_readmitted_risk) > 1:

            t_stat_risk, p_value_risk = ttest_ind(
                readmitted_risk,
                not_readmitted_risk,
                equal_var=False
            )

            mean_difference_risk = (
                readmitted_risk.mean()
                - not_readmitted_risk.mean()
            )

            st.write(
                f"**T-statistic:** {t_stat_risk:.4f}"
            )

            st.write(
                f"**P-value:** {p_value_risk:.4e}"
            )

            st.write(
                f"**Mean difference:** "
                f"{mean_difference_risk:.4f}"
            )

            if p_value_risk < 0.05:
                st.success(
                    "Statistically significant at α = 0.05."
                )
            else:
                st.info(
                    "Not statistically significant at α = 0.05."
                )


        st.markdown("---")


        st.subheader(
            "2. Readmission vs Treatment Cost"
        )

        readmitted_cost = filtered_df[
            filtered_df["readmission"] == "Yes"
        ]["treatment_cost"].dropna()

        not_readmitted_cost = filtered_df[
            filtered_df["readmission"] == "No"
        ]["treatment_cost"].dropna()

        if len(readmitted_cost) > 1 and len(not_readmitted_cost) > 1:

            t_stat_cost, p_value_cost = ttest_ind(
                readmitted_cost,
                not_readmitted_cost,
                equal_var=False
            )

            mean_difference_cost = (
                readmitted_cost.mean()
                - not_readmitted_cost.mean()
            )

            st.write(
                f"**T-statistic:** {t_stat_cost:.4f}"
            )

            st.write(
                f"**P-value:** {p_value_cost:.4e}"
            )

            st.write(
                f"**Mean cost difference:** "
                f"₹{mean_difference_cost:,.2f}"
            )

            if p_value_cost < 0.05:
                st.success(
                    "Statistically significant at α = 0.05."
                )
            else:
                st.info(
                    "Not statistically significant at α = 0.05."
                )


        st.markdown("---")


        st.subheader(
            "3. Risk Score Across Diseases"
        )

        disease_groups = [
            group["risk_score"].dropna()
            for _, group
            in filtered_df.groupby("disease")
        ]

        disease_groups = [
            group
            for group in disease_groups
            if len(group) > 1
        ]

        if len(disease_groups) >= 2:

            f_stat_disease, p_value_disease = f_oneway(
                *disease_groups
            )

            st.write(
                f"**F-statistic:** "
                f"{f_stat_disease:.4f}"
            )

            st.write(
                f"**P-value:** "
                f"{p_value_disease:.4e}"
            )

            if p_value_disease < 0.05:
                st.success(
                    "Risk scores differ significantly "
                    "across disease groups."
                )
            else:
                st.info(
                    "No statistically significant "
                    "difference detected."
                )


        st.markdown("---")


        st.subheader(
            "4. Treatment Cost Across Admission Types"
        )

        admission_groups = [
            group["treatment_cost"].dropna()
            for _, group
            in filtered_df.groupby("admission_type")
        ]

        admission_groups = [
            group
            for group in admission_groups
            if len(group) > 1
        ]

        if len(admission_groups) >= 2:

            f_stat_admission, p_value_admission = f_oneway(
                *admission_groups
            )

            st.write(
                f"**F-statistic:** "
                f"{f_stat_admission:.4f}"
            )

            st.write(
                f"**P-value:** "
                f"{p_value_admission:.4e}"
            )

            if p_value_admission < 0.05:
                st.success(
                    "Treatment costs differ significantly "
                    "across admission types."
                )
            else:
                st.info(
                    "No statistically significant "
                    "difference detected."
                )


        st.markdown("---")


        st.subheader(
            "5. Admission Type vs Readmission"
        )

        contingency_table = pd.crosstab(
            filtered_df["admission_type"],
            filtered_df["readmission"]
        )

        chi2, p_value_chi, dof, expected = (
            chi2_contingency(
                contingency_table
            )
        )

        st.dataframe(
            contingency_table
        )

        st.write(
            f"**Chi-square statistic:** "
            f"{chi2:.4f}"
        )

        st.write(
            f"**Degrees of freedom:** "
            f"{dof}"
        )

        st.write(
            f"**P-value:** "
            f"{p_value_chi:.4e}"
        )

        if p_value_chi < 0.05:
            st.success(
                "There is a statistically significant "
                "association between admission type "
                "and readmission."
            )
        else:
            st.info(
                "No statistically significant association "
                "was detected."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
        font-size:13px;
    ">
        Healthcare Patient Analytics & Risk Analysis System
        <br>
        Built with Python • Pandas • NumPy • Matplotlib • Seaborn • SciPy • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)