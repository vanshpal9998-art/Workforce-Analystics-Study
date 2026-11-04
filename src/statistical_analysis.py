import pandas as pd
import numpy as np
from statistics import NormalDist

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/employee_data_cleaned.csv"
)

# Select numerical columns
numerical_df = df.select_dtypes(include="number")

# Descriptive statistics
print("\n--- DESCRIPTIVE STATISTICS ---")

print(numerical_df.describe())

# Mean
print("\nMean:")
print(numerical_df.mean())

# Median
print("\nMedian:")
print(numerical_df.median())

# Mode
print("\nMode:")
print(numerical_df.mode().iloc[0])

# Variance
print("\nVariance:")
print(numerical_df.var())

# Standard deviation
print("\nStandard Deviation:")
print(numerical_df.std())

print("\n--- ATTRITION GROUP ANALYSIS ---")

group_analysis = df.groupby("Attrition").agg(
    Average_Age=("Age", "mean"),
    Average_Income=("MonthlyIncome", "mean"),
    Average_Experience=("YearsAtCompany", "mean"),
    Average_Satisfaction=("JobSatisfaction", "mean"),
    Employee_Count=("Attrition", "count")
)

print(group_analysis.round(2))

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns
numerical_df = df.select_dtypes(include="number")

# Calculate correlation
correlation = numerical_df.corr()

print("\n--- CORRELATION MATRIX ---")

print(correlation.round(2))

# Create heatmap
plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Employee Workforce Correlation Heatmap")

plt.tight_layout()

plt.savefig("reports/correlation_heatmap.png")

plt.show()

from scipy.stats import ttest_ind

# Separate employee groups
left_employees = df[
    df["Attrition"] == "Yes"
]["Age"]

stayed_employees = df[
    df["Attrition"] == "No"
]["Age"]

# Perform Welch's independent t-test
t_stat, p_value = ttest_ind(
    left_employees,
    stayed_employees,
    equal_var=False
)

print("\n--- AGE HYPOTHESIS TEST ---")

print("T-statistic:", t_stat)

print("P-value:", p_value)

if p_value < 0.05:
    print("Reject the Null Hypothesis")
    print("Evidence of a difference in average age.")
else:
    print("Fail to Reject the Null Hypothesis")
    print("Insufficient evidence of a difference in average age.")
    
from scipy.stats import chi2_contingency

# Create contingency table
contingency_table = pd.crosstab(
    df["OverTime"],
    df["Attrition"]
)

print("\n--- CONTINGENCY TABLE ---")

print(contingency_table)

# Chi-square test
chi2, p_value, dof, expected = chi2_contingency(
    contingency_table
)

print("\nChi-square Statistic:", chi2)

print("P-value:", p_value)

print("Degrees of Freedom:", dof)

if p_value < 0.05:
    print("Overtime and Attrition are statistically associated.")
else:
    print("Insufficient evidence of an association.")
    
    
department_attrition = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\n--- DEPARTMENT ATTRITION RATE ---")

print(department_attrition.round(2))
    
from pathlib import Path

Path("reports").mkdir(
    parents=True,
    exist_ok=True
)

group_analysis.to_csv(
    "reports/attrition_group_statistics.csv"
)

correlation.to_csv(
    "reports/correlation_matrix.csv"
)

department_attrition.to_csv(
    "reports/department_attrition_rates.csv"
)

print("\nStatistical reports saved successfully!")