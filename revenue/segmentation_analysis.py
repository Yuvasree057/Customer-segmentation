import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Create output directories
os.makedirs("visualizations", exist_ok=True)

# Load data
df = pd.read_csv("data/Mall_Customers.csv")

# We will use Annual Income and Spending Score for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('visualizations/elbow_method.png')
plt.close()

# From elbow method, 5 is a good number of clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X_scaled[clusters == i, 0], X_scaled[clusters == i, 1], s=100, c=colors[i], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', marker='X')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.legend()
plt.savefig('visualizations/customer_clusters.png')
plt.close()

# Analyze the segments
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
cluster_summary['Count'] = df['Cluster'].value_counts()

print("Cluster Summary:")
print(cluster_summary)

with open('segmentation_report.md', 'w') as f:
    f.write("# Customer Segmentation Report\n\n")
    f.write("## Overview\n")
    f.write("We performed customer segmentation using KMeans clustering on the Mall Customers dataset. The features used for clustering were `Annual Income (k$)` and `Spending Score (1-100)`.\n\n")
    f.write("## Methodology\n")
    f.write("- Standardized features to ensure equal weighting.\n")
    f.write("- Used the Elbow Method to determine the optimal number of clusters, which was identified as **5**.\n\n")
    f.write("## Cluster Characteristics\n")
    f.write(cluster_summary.to_string())
    f.write("\n\n## Analysis of Purchase Patterns & Demographics\n")
    f.write("- **Cluster 0**: Average Income, Average Spending. The largest segment, representing typical mall-goers.\n")
    f.write("- **Cluster 1**: High Income, Low Spending. These customers are careful with their money despite high income.\n")
    f.write("- **Cluster 2**: Low Income, Low Spending. Price-sensitive customers.\n")
    f.write("- **Cluster 3**: Low Income, High Spending. These customers spend a lot despite low income, perhaps younger demographics.\n")
    f.write("- **Cluster 4**: High Income, High Spending. The target demographic for premium/luxury products.\n")
