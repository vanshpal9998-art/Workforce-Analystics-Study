import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/employee_data_cleaned.csv"
)

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

# Set visualization style
sns.set_theme(style="whitegrid")

total_employees = len(df)

attrition_count = (
    df["Attrition"].eq("Yes").sum()
)

active_employees = (
    df["Attrition"].eq("No").sum()
)

attrition_rate = (
    attrition_count / total_employees
) * 100

retention_rate = (
    active_employees / total_employees
) * 100

print("\n--- WORKFORCE OVERVIEW ---")

print("Total Employees:", total_employees)

print("Employees Left:", attrition_count)

print("Active Employees:", active_employees)

print("Attrition Label Share:", round(attrition_rate, 2), "%")

print("Retention Label Share:", round(retention_rate, 2), "%")

department_counts = df["Department"].value_counts()

print("\nDepartment Distribution:")
print(department_counts)

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Department",
    order=department_counts.index
)

plt.title("Employee Distribution by Department")

plt.xlabel("Department")

plt.ylabel("Number of Employees")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("reports/department_distribution.png")

plt.show()

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True
)

plt.title("Employee Age Distribution")

plt.xlabel("Age")

plt.ylabel("Employee Count")

plt.tight_layout()

plt.savefig("reports/age_distribution.png")

plt.show()

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="MonthlyIncome",
    bins=30,
    kde=True
)

plt.title("Monthly Salary Distribution")

plt.xlabel("Monthly Income")

plt.ylabel("Employee Count")

plt.tight_layout()

plt.savefig("reports/salary_distribution.png")

plt.show()

average_salary = df["MonthlyIncome"].mean()

median_salary = df["MonthlyIncome"].median()

print("\nAverage Monthly Salary:", round(average_salary, 2))

print("Median Monthly Salary:", round(median_salary, 2))

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Attrition"
)

plt.title("Employee Attrition Distribution")

plt.xlabel("Attrition Status")

plt.ylabel("Employee Count")

plt.tight_layout()

plt.savefig("reports/attrition_distribution.png")

plt.show()

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)

plt.title("Attrition by Department")

plt.xlabel("Department")

plt.ylabel("Employee Count")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("reports/attrition_by_department.png")

plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title("Overtime vs Employee Attrition")

plt.xlabel("Overtime")

plt.ylabel("Employee Count")

plt.tight_layout()

plt.savefig("reports/overtime_attrition.png")

plt.show()

print("\n========== EDA SUMMARY ==========")

print("Total Employees:", len(df))

print("Average Age:", round(df["Age"].mean(), 2))

print(
    "Average Monthly Income:",
    round(df["MonthlyIncome"].mean(), 2)
)

print(
    "Average Years at Company:",
    round(df["YearsAtCompany"].mean(), 2)
)

print(
    "Highest Employee Department:",
    df["Department"].value_counts().idxmax()
)

print(
    "Most Common Job Role:",
    df["JobRole"].value_counts().idxmax()
)

print("=================================")