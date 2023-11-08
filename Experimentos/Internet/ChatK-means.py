from sklearn.cluster import KMeans
import numpy as np

# Generate some random data for demonstration purposes
data = np.random.rand(100, 2)

# Specify the number of clusters (K)
K = 3

# Create a K-Means instance
kmeans = KMeans(n_clusters=K)

# Fit the model to the data
kmeans.fit(data)

# Get cluster assignments
labels = kmeans.labels_

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Predict cluster assignment for a new data point
new_data_point = np.array([[0.5, 0.5]])
predicted_cluster = kmeans.predict(new_data_point)

print("Cluster Assignments:", labels)
print("Cluster Centers:", cluster_centers)
print("Predicted Cluster for New Data Point:", predicted_cluster)