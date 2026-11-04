import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.xls"
)

# 1. Display data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# 2. Count numerical columns
print("\n--- NUMERICAL COLUMNS ---")
print(df.select_dtypes(include=["number"]).columns.tolist())

# 3. Count categorical columns
print("\n--- CATEGORICAL COLUMNS ---")
print(df.select_dtypes(include=["object"]).columns.tolist())

# 4. Unique values in categorical columns
print("\n--- UNIQUE VALUES ---")

for column in df.select_dtypes(include=["object"]).columns:
    print(f"\n{column}:")
    print(df[column].unique())

# 5. Number of unique values
print("\n--- UNIQUE VALUE COUNTS ---")
print(df.nunique())

print("\n--- CONSTANT COLUMNS ---")

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

print(constant_columns)

print("\n--- IDENTIFIER COLUMNS ---")

possible_ids = [
    column for column in df.columns
    if "id" in column.lower() or "number" in column.lower()
]

print(possible_ids)

print("\n--- DATA QUALITY CHECK ---")

print("Total missing values:", df.isnull().sum().sum())

print("Total duplicate rows:", df.duplicated().sum())

print("Minimum employee age:", df["Age"].min())

print("Maximum employee age:", df["Age"].max())

print("Minimum monthly income:", df["MonthlyIncome"].min())

print("Maximum monthly income:", df["MonthlyIncome"].max())

print("Attrition categories:", df["Attrition"].unique())