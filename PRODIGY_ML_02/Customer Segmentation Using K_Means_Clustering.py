import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r'/Users/aishwarya/Jain University/Masters/Prodigy InfoTech/PRODIGY_ML_02/Mall_Customers.csv')

print(df)

print(df.head()) 

# Assuming 'Annual Income' and 'Spending Score' are relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
k = 2  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X_scaled)

# Add the 'Cluster' column to the DataFrame
df['Cluster'] = kmeans.labels_

# Visualizations

# Cluster Distribution Plot
cluster_counts = df.groupby('Cluster')['CustomerID'].count()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Distribution')
plt.show()

# Plot cluster scatter plot with color-coding by gender
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Gender'].map({'Male': 0, 'Female': 1}), cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Cluster Scatter Plot with Color-Coding by Gender')
plt.colorbar(label='Gender')
plt.show()

# Plot cluster scatter plot with centroids
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Cluster Scatter Plot with Centroids')
plt.legend()
plt.show()

# Plot cluster boxplot comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df)
plt.xlabel('Cluster')
plt.ylabel('Annual Income (k$)')
plt.title('Cluster Boxplot: Annual Income')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df)
plt.xlabel('Cluster')
plt.ylabel('Spending Score (1-100)')
plt.title('Cluster Boxplot: Spending Score')
plt.show()

# Plot cluster pairplot
sns.pairplot(df, hue='Cluster', vars=['Annual Income (k$)', 'Spending Score (1-100)', 'Age'], palette='viridis')
plt.title('Cluster Pairplot')
plt.show()

# Calculate average feature values by cluster
cluster_means = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean()

# Plot cluster radar chart
labels = np.array(['Annual Income', 'Spending Score', 'Age'])
stats_cluster_0 = cluster_means.loc[cluster_means.index[0]].values
stats_cluster_0 = np.concatenate((stats_cluster_0, [stats_cluster_0[0]]))  # Close the loop
stats_cluster_1 = cluster_means.loc[cluster_means.index[1]].values
stats_cluster_1 = np.concatenate((stats_cluster_1, [stats_cluster_1[0]]))  # Close the loop

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # Add the first angle at the end to close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, stats_cluster_0, color='b', alpha=0.25)
ax.plot(angles, stats_cluster_0, color='b', linewidth=2, label='Cluster 0')
ax.fill(angles, stats_cluster_1, color='r', alpha=0.25)
ax.plot(angles, stats_cluster_1, color='r', linewidth=2, label='Cluster 1')

ax.set_yticklabels([])
plt.xticks(angles[:-1], labels, color='black', size=12)
plt.title('Cluster Radar Chart', size=20, color='black', y=1.1)
plt.legend()
plt.show()
