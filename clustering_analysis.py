import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to perform clustering analysis
def clustering_analysis(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Prepare the features for clustering (excluding the target variable)
    X = data.drop('p401', axis=1)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=6, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
    print(f"Silhouette Score for KMeans: {silhouette_kmeans:.2f}")

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Filtering out noise points labeled as -1 by DBSCAN for silhouette score
    valid_dbscan_labels = dbscan_labels[dbscan_labels != -1]
    valid_dbscan_data = X_scaled[dbscan_labels != -1]

    if len(valid_dbscan_labels) > 0:
        silhouette_dbscan = silhouette_score(valid_dbscan_data, valid_dbscan_labels)
        print(f"Silhouette Score for DBSCAN: {silhouette_dbscan:.2f}")
    else:
        print("DBSCAN did not find any clusters, all points labeled as noise.")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plotting KMeans Clustering Results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='viridis', s=50)
    plt.title('KMeans Clustering')

    # Plotting DBSCAN Clustering Results
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='viridis', s=50)
    plt.title('DBSCAN Clustering')

    plt.tight_layout()
    plt.show()

# Define the path to your dataset
file_path = '401k_data.csv'  # Update the path if the file is located elsewhere

# Call the clustering analysis function
clustering_analysis(file_path)
