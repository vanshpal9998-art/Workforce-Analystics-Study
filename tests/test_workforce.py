import unittest
import pandas as pd
from pathlib import Path


# --------------------------------
# 1. Load Dataset
# --------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "employee_data_cleaned.csv"
)


def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# --------------------------------
# 2. Testing Class
# --------------------------------

class TestWorkforceAnalytics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.df = load_dataset()


    # --------------------------------
    # Test 1: Dataset Not Empty
    # --------------------------------

    def test_dataset_not_empty(self):

        self.assertGreater(
            len(self.df),
            0,
            "Dataset is empty"
        )


    # --------------------------------
    # Test 2: Required Columns
    # --------------------------------

    def test_required_columns(self):

        required_columns = [
            "Attrition",
            "Department",
            "MonthlyIncome"
        ]

        for column in required_columns:

            self.assertIn(
                column,
                self.df.columns
            )


    # --------------------------------
    # Test 3: Missing Values
    # --------------------------------

    def test_no_missing_values(self):

        missing_values = self.df.isnull().sum().sum()

        self.assertEqual(
            missing_values,
            0,
            "Dataset contains missing values"
        )


    # --------------------------------
    # Test 4: Duplicate Records
    # --------------------------------

    def test_no_duplicate_rows(self):

        duplicates = self.df.duplicated().sum()

        self.assertEqual(
            duplicates,
            0,
            "Dataset contains duplicate rows"
        )


    # --------------------------------
    # Test 5: Attrition Categories
    # --------------------------------

    def test_attrition_values(self):

        valid_values = {"Yes", "No"}

        actual_values = set(
            self.df["Attrition"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        self.assertTrue(
            actual_values.issubset(valid_values),
            "Invalid attrition values found"
        )


    # --------------------------------
    # Test 6: Valid Age
    # --------------------------------

    def test_valid_age(self):

        if "Age" in self.df.columns:

            self.assertTrue(
                self.df["Age"].between(18, 100).all(),
                "Age contains invalid values"
            )


    # --------------------------------
    # Test 7: Valid Salary
    # --------------------------------

    def test_valid_salary(self):

        self.assertTrue(
            (
                self.df["MonthlyIncome"] >= 0
            ).all(),
            "Salary contains negative values"
        )


    # --------------------------------
    # Test 8: Attrition KPI
    # --------------------------------

    def test_attrition_calculation(self):

        total = len(self.df)

        left = (
            self.df["Attrition"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
            .sum()
        )

        attrition_share = (
            left / total
        ) * 100

        self.assertGreaterEqual(
            attrition_share,
            0
        )

        self.assertLessEqual(
            attrition_share,
            100
        )


# --------------------------------
# 3. Run Tests
# --------------------------------

if __name__ == "__main__":

    unittest.main(verbosity=2)
    
def calculate_attrition(left, total):

    if total <= 0:
        raise ValueError("Total employees must be positive")

    return (left / total) * 100


print(calculate_attrition(20, 100))

assert 10 > 5