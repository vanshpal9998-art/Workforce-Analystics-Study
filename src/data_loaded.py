import pandas as pd

# Load dataset
file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.xls"

df = pd.read_csv(file_path)

# Display first five records
print("\nFirst Five Records:")
print(df.head())

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Attrition distribution
print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())

# Attrition percentages
print("\nAttrition Percentage:")
print(df["Attrition"].value_counts(normalize=True) * 100)