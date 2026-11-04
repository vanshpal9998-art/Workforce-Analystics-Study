import pandas as pd
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/employee_data_cleaned.csv"
)

print("Dataset loaded successfully!")

# Total employees
total_employees = len(df)

# Employees who left
attrition_count = (
    df["Attrition"].eq("Yes").sum()
)

# Employees who stayed
active_employees = (
    df["Attrition"].eq("No").sum()
)

# Attrition label share
attrition_rate = (
    attrition_count / total_employees
) * 100

# Retention label share
retention_rate = (
    active_employees / total_employees
) * 100

print("\n========== WORKFORCE KPIs ==========")

print("Total Employees:", total_employees)

print("Employees Left:", attrition_count)

print("Employees Stayed:", active_employees)

print("Attrition Label Share:", round(attrition_rate, 2), "%")

print("Retention Label Share:", round(retention_rate, 2), "%")

department_analysis = df.groupby(
    "Department"
).agg(
    Total_Employees=("Attrition", "count"),
    Employees_Left=(
        "Attrition",
        lambda x: (x == "Yes").sum()
    )
)

department_analysis["Attrition_Percentage"] = (
    department_analysis["Employees_Left"]
    / department_analysis["Total_Employees"]
) * 100

department_analysis = department_analysis.sort_values(
    by="Attrition_Percentage",
    ascending=False
)

print("\n========== DEPARTMENT ANALYSIS ==========")

print(department_analysis.round(2))

highest_attrition_department = (
    department_analysis["Attrition_Percentage"].idxmax()
)

highest_attrition_value = (
    department_analysis["Attrition_Percentage"].max()
)

print("\nDepartment with Highest Attrition:")

print(highest_attrition_department)

print(
    "Attrition Percentage:",
    round(highest_attrition_value, 2),
    "%"
)

group_analysis = df.groupby("Attrition").agg(
    Average_Age=("Age", "mean"),
    Average_Salary=("MonthlyIncome", "mean"),
    Average_Experience=("YearsAtCompany", "mean"),
    Average_Job_Satisfaction=("JobSatisfaction", "mean")
)

print("\n========== ATTRITION GROUP COMPARISON ==========")

print(group_analysis.round(2))

overtime_analysis = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print("\n========== OVERTIME ANALYSIS ==========")

print(overtime_analysis.round(2))

insights = []

# Insight 1: Overall workforce
insights.append(
    f"The organization has {total_employees} employee records."
)

# Insight 2: Attrition
insights.append(
    f"{attrition_count} employees are marked as having left."
)

# Insight 3: Attrition percentage
insights.append(
    f"The overall attrition label share is {attrition_rate:.2f}%."
)

# Insight 4: Department
insights.append(
    f"{highest_attrition_department} has the highest "
    f"department attrition percentage "
    f"({highest_attrition_value:.2f}%)."
)

# Display insights
print("\n========== BUSINESS INSIGHTS ==========")

for number, insight in enumerate(insights, start=1):
    print(f"{number}. {insight}")
    
# Create reports directory
Path("reports").mkdir(
    parents=True,
    exist_ok=True
)

# Save insights
with open(
    "reports/business_insights.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("WORKFORCE ANALYTICS BUSINESS INSIGHTS\n")

    file.write("=" * 45 + "\n\n")

    file.write(f"Total Employees: {total_employees}\n")

    file.write(f"Employees Left: {attrition_count}\n")

    file.write(
        f"Attrition Label Share: {attrition_rate:.2f}%\n\n"
    )

    file.write("KEY FINDINGS:\n")

    for number, insight in enumerate(insights, start=1):
        file.write(f"{number}. {insight}\n")

    file.write("\nDEPARTMENT ANALYSIS:\n")

    file.write(
        department_analysis.round(2).to_string()
    )

    file.write("\n\nOVERTIME ANALYSIS:\n")

    file.write(
        overtime_analysis.round(2).to_string()
    )

print("\nBusiness Insights Report Saved Successfully!")