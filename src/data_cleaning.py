import pandas as pd
from pathlib import Path

# --------------------------------
# 1. Load Original Dataset
# --------------------------------

input_path = Path(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.xls"
)

output_path = Path(
    "data/processed/employee_data_cleaned.csv"
)

report_path = Path(
    "reports/data_quality_report.txt"
)

df = pd.read_csv(input_path)

# Preserve original data
original_df = df.copy()

print("Original Dataset Shape:", df.shape)


# --------------------------------
# 2. Check Missing Values
# --------------------------------

print("\nMissing Values Before Cleaning:")

missing_before = df.isnull().sum()

print(missing_before[missing_before > 0])


# --------------------------------
# 3. Check Duplicate Records
# --------------------------------

duplicates_before = df.duplicated().sum()

print("\nDuplicate Records:", duplicates_before)

# Remove exact duplicate rows
df = df.drop_duplicates()


# --------------------------------
# 4. Check Data Types
# --------------------------------

print("\nData Types Before Cleaning:")

print(df.dtypes)


# --------------------------------
# 5. Clean Categorical Text
# --------------------------------

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    df[column] = df[column].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )


# --------------------------------
# 6. Check Important Values
# --------------------------------

print("\nAge Range:")

print(df["Age"].min(), df["Age"].max())

print("\nMonthly Income Range:")

print(
    df["MonthlyIncome"].min(),
    df["MonthlyIncome"].max()
)

print("\nAttrition Categories:")

print(df["Attrition"].value_counts(dropna=False))


# --------------------------------
# 7. Identify Constant Columns
# --------------------------------

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

print("\nConstant Columns:")

print(constant_columns)


# --------------------------------
# 8. Remove Uninformative Columns
# --------------------------------

df = df.drop(
    columns=constant_columns,
    errors="ignore"
)


# --------------------------------
# 9. Final Quality Checks
# --------------------------------

missing_after = df.isnull().sum().sum()

duplicates_after = df.duplicated().sum()

print("\nFinal Dataset Shape:", df.shape)

print("Missing Values After Cleaning:", missing_after)

print("Duplicate Rows After Cleaning:", duplicates_after)


# --------------------------------
# 10. Save Cleaned Dataset
# --------------------------------

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(output_path, index=False)

print("\nCleaned Dataset Saved Successfully!")


# --------------------------------
# 11. Generate Quality Report
# --------------------------------

report_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("WORKFORCE ANALYTICS - DATA QUALITY REPORT\n")

    file.write("\nOriginal Dataset Shape: ")
    file.write(str(original_df.shape))

    file.write("\nFinal Dataset Shape: ")
    file.write(str(df.shape))

    file.write("\nDuplicates Before Cleaning: ")
    file.write(str(duplicates_before))

    file.write("\nDuplicates After Cleaning: ")
    file.write(str(duplicates_after))

    file.write("\nMissing Values Before Cleaning: ")
    file.write(str(int(missing_before.sum())))

    file.write("\nMissing Values After Cleaning: ")
    file.write(str(int(missing_after)))

    file.write("\nRemoved Constant Columns: ")
    file.write(str(constant_columns))

print("Quality Report Generated Successfully!")