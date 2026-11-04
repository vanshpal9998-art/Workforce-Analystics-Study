import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------
# 1. Project Configuration
# --------------------------------

st.set_page_config(
    page_title="Workforce Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "employee_data_cleaned.csv"

SEGMENT_PATH = BASE_DIR / "data" / "processed" / "employee_segments.csv"

MODEL_PATH = BASE_DIR / "reports" / "model_comparison.csv"


# --------------------------------
# 2. Dashboard Title
# --------------------------------

st.title("📊 Workforce Analytics Dashboard")

st.write(
    "Analyze employee demographics, salary distribution, "
    "department performance indicators and attrition patterns."
)

st.divider()


# --------------------------------
# 3. Load Dataset
# --------------------------------

@st.cache_data
def load_data(path):

    return pd.read_csv(path)


if not DATA_PATH.exists():

    st.error(
        "Cleaned dataset not found. "
        "Please run your data cleaning script first."
    )

    st.stop()


df = load_data(str(DATA_PATH))


# --------------------------------
# 4. Validate Required Columns
# --------------------------------

required_columns = [
    "Attrition",
    "Department",
    "MonthlyIncome"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(f"Missing required columns: {missing_columns}")

    st.stop()


# --------------------------------
# 5. Sidebar Filters
# --------------------------------

st.sidebar.header("Dashboard Filters")

departments = sorted(
    df["Department"].dropna().unique().tolist()
)

selected_departments = st.sidebar.multiselect(
    "Select Departments",
    departments,
    default=departments
)

filtered_df = df[
    df["Department"].isin(selected_departments)
].copy()


# --------------------------------
# 6. Calculate KPIs
# --------------------------------

total_employees = len(filtered_df)

attrition_count = (
    filtered_df["Attrition"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("yes")
    .sum()
)

active_employees = total_employees - attrition_count

attrition_share = (
    (attrition_count / total_employees) * 100
    if total_employees > 0
    else 0
)


# --------------------------------
# 7. Display KPI Cards
# --------------------------------

st.subheader("Workforce Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Employees",
    f"{total_employees:,}"
)

col2.metric(
    "Attrition Count",
    f"{attrition_count:,}"
)

col3.metric(
    "Active Employees",
    f"{active_employees:,}"
)

col4.metric(
    "Attrition Label Share",
    f"{attrition_share:.2f}%"
)

st.divider()


# --------------------------------
# 8. Attrition Distribution
# --------------------------------

st.subheader("Employee Attrition Distribution")

attrition_data = (
    filtered_df["Attrition"]
    .value_counts()
    .reset_index()
)

attrition_data.columns = ["Attrition", "Count"]

fig1 = px.pie(
    attrition_data,
    names="Attrition",
    values="Count",
    title="Employees Who Left vs Stayed",
    hole=0.4
)

st.plotly_chart(fig1, use_container_width=True)


# --------------------------------
# 9. Department Distribution
# --------------------------------

st.subheader("Department-wise Employee Distribution")

department_data = (
    filtered_df["Department"]
    .value_counts()
    .reset_index()
)

department_data.columns = ["Department", "Employees"]

fig2 = px.bar(
    department_data,
    x="Department",
    y="Employees",
    title="Employees by Department",
    text="Employees"
)

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------
# 10. Department-wise Attrition
# --------------------------------

st.subheader("Department-wise Attrition Share")

department_attrition = (
    filtered_df.assign(
        Left=filtered_df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    )
    .groupby("Department", observed=True)
    .agg(
        Employees=("Left", "size"),
        Attrition_Count=("Left", "sum")
    )
    .reset_index()
)

department_attrition["Attrition_Percentage"] = (
    department_attrition["Attrition_Count"]
    / department_attrition["Employees"]
    * 100
)

fig3 = px.bar(
    department_attrition,
    x="Department",
    y="Attrition_Percentage",
    title="Attrition Label Share by Department",
    text_auto=".2f"
)

st.plotly_chart(fig3, use_container_width=True)


# --------------------------------
# 11. Salary Distribution
# --------------------------------

st.subheader("Employee Salary Distribution")

fig4 = px.histogram(
    filtered_df,
    x="MonthlyIncome",
    nbins=30,
    title="Monthly Salary Distribution"
)

st.plotly_chart(fig4, use_container_width=True)


# --------------------------------
# 12. Overtime Analysis
# --------------------------------

if "OverTime" in filtered_df.columns:

    st.subheader("Overtime and Attrition Analysis")

    overtime_data = pd.crosstab(
        filtered_df["OverTime"],
        filtered_df["Attrition"]
    )

    fig5 = px.bar(
        overtime_data,
        barmode="group",
        title="Overtime vs Attrition"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


# --------------------------------
# 13. Employee Segmentation
# --------------------------------

if SEGMENT_PATH.exists():

    st.subheader("Employee Segmentation")

    segment_df = pd.read_csv(SEGMENT_PATH)

    if "Cluster" in segment_df.columns:

        cluster_data = (
            segment_df["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        cluster_data.columns = ["Cluster", "Employees"]

        fig6 = px.bar(
            cluster_data,
            x="Cluster",
            y="Employees",
            title="Employees in Each Cluster"
        )

        st.plotly_chart(
            fig6,
            use_container_width=True
        )


# --------------------------------
# 14. ML Model Comparison
# --------------------------------

if MODEL_PATH.exists():

    st.subheader("Machine Learning Model Comparison")

    model_df = pd.read_csv(MODEL_PATH)

    st.dataframe(
        model_df,
        use_container_width=True
    )


# --------------------------------
# 15. Employee Data Table
# --------------------------------

st.subheader("Employee Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)


# --------------------------------
# 16. Download Filtered Data
# --------------------------------

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Employee Data",
    data=csv_data,
    file_name="filtered_employee_data.csv",
    mime="text/csv"
)


# --------------------------------
# 17. Dashboard Footer
# --------------------------------

st.divider()

st.caption(
    "Workforce Analytics & Employee Attrition Intelligence System | "
    "Educational analysis using historical workforce data."
)

st.caption(
    "Important: Attrition label share is not an annual employee turnover rate. "
    "The dataset does not establish why an employee left."
)