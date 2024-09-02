import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
# Assuming you have a CSV file for simplicity
data = pd.read_csv('raw_data.csv')

# Preprocessing
# Select relevant features for clustering
features = data[["Marital status","Age", "Education"]]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Implement K-Means
num_clusters = 3  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Evaluate clustering
silhouette_avg = silhouette_score(scaled_features, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Save the clustered data to a new CSV
data.to_csv('clustered_data.csv', index=False)
