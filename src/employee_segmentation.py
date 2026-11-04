import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv(
    "data/processed/employee_data_cleaned.csv"
)

print("Dataset Shape:", df.shape)
print(df.head())

features = [
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "JobSatisfaction",
    "WorkLifeBalance",
    "JobInvolvement",
    "TrainingTimesLastYear"
]

# Verify that all selected columns exist
missing_columns = [
    col for col in features if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

X = df[features].copy()

print("\nSelected Features:")
print(X.head())

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

print("\nMissing values handled successfully!")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("\nFeature Scaling Completed!")

inertia_values = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.tight_layout()

Path("reports").mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig("reports/elbow_method.png")

plt.show()

silhouette_scores = {}

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores[k] = score

print("\nSilhouette Scores:")

for k, score in silhouette_scores.items():

    print(
        f"K = {k}, Score = {score:.4f}"
    )
    
best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

print("\nSelected Number of Clusters:", best_k)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustering Completed!")

print(df[["Cluster"]].value_counts().sort_index())

cluster_profile = df.groupby("Cluster")[features].mean()

print("\n========== CLUSTER PROFILE ==========")

print(cluster_profile.round(2))

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="YearsAtCompany",
    y="MonthlyIncome",
    hue="Cluster",
    palette="viridis",
    s=70
)

plt.title("Employee Segmentation")

plt.xlabel("Years at Company")

plt.ylabel("Monthly Income")

plt.tight_layout()

plt.savefig(
    "reports/employee_clusters.png"
)

plt.show()

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\n========== CLUSTER SIZES ==========")

print(cluster_counts)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Cluster",
    hue="Cluster",
    palette="viridis",
    legend=False
)

plt.title("Number of Employees in Each Cluster")

plt.xlabel("Cluster")

plt.ylabel("Employee Count")

plt.tight_layout()

plt.savefig(
    "reports/cluster_distribution.png"
)

plt.show()

# Save employee records with cluster labels
df.to_csv(
    "data/processed/employee_segments.csv",
    index=False
)

# Save cluster profiles
cluster_profile.to_csv(
    "reports/cluster_profiles.csv"
)

# Save cluster sizes
cluster_counts.to_csv(
    "reports/cluster_sizes.csv"
)

print("\nSegmentation Results Saved Successfully!")

import joblib

joblib.dump(
    {
        "features": features,
        "imputer": imputer,
        "scaler": scaler,
        "kmeans": kmeans
    },
    "reports/employee_segmentation_model.pkl"
)

print("Segmentation Model Saved Successfully!")

