import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/processed/employee_data_cleaned.csv"
)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================
# 2. SELECT TARGET AND FEATURES
# ==========================================

# Convert target labels into numerical values
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Check target values
if df["Attrition"].isnull().any():
    raise ValueError(
        "Unexpected or missing values found in Attrition."
    )

# Remove identifier and uninformative columns if present
columns_to_drop = [
    "EmployeeNumber",
    "EmployeeID",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# Separate features and target
X = df.drop(columns=["Attrition"])

y = df["Attrition"]

print("\nFeatures Shape:", X.shape)

print("Target Distribution:")
print(y.value_counts())


# ==========================================
# 3. IDENTIFY COLUMN TYPES
# ==========================================

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))

print("Testing Records:", len(X_test))


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ==========================================
# 6. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

}


# ==========================================
# 7. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict labels
    y_pred = pipeline.predict(X_test)

    # Predict probability of attrition
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    trained_models[model_name] = pipeline

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Stayed", "Left"],
            zero_division=0
        )
    )


# ==========================================
# 8. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.round(3))


# ==========================================
# 9. SELECT MODEL
# ==========================================

best_model_name = results_df.loc[
    results_df["ROC AUC"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nSelected Model:", best_model_name)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions
)

Path("reports").mkdir(
    parents=True,
    exist_ok=True
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"]
)

plt.title("Employee Attrition Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png"
)

plt.show()


# ==========================================
# 11. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

joblib.dump(
    best_model,
    "reports/best_attrition_model.pkl"
)

print("\nModel Comparison Saved!")

print("Best Model Saved Successfully!")

print("Selected Model:", best_model_name)

Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

