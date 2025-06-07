import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
file_path = r"C:\Users\VARUN\Desktop\Python\ML\my_csv.csv"
df = pd.read_csv(file_path)

# Drop categorical columns and rename duplicate column
df_clean = df.drop(columns=["Gender", "Learner"])
df_clean = df.drop(columns=[ "I understand things better in class when I participate in role-playing..1" ])

# Normalize numerical features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean)

# Determine optimal number of clusters using Elbow Method
inertia = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal K")
plt.show()

# Choose optimal K (e.g., 4 based on elbow and silhouette score)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

df["Cluster"] = clusters

# Compute cluster profiles using only numerical data
numerical_columns = df_clean.columns
df_numerical = df[numerical_columns]
df_numerical["Cluster"] = clusters
cluster_profiles = df_numerical.groupby("Cluster").mean()
print(cluster_profiles.T)

# Compute final performance score
final_silhouette = silhouette_score(df_scaled, clusters)
print(f"Final Silhouette Score: {final_silhouette:.4f}")
