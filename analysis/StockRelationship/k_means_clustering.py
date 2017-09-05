from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_means_clustering(x_9d):
    kmeans = KMeans(n_clusters=3)

    # Compute cluster centers and predict cluster indices
    X_clustered = kmeans.fit_predict(x_9d)

    # Define out own color map
    LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b'}
    label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

    # Plot the scatter digram
    plt.figure(figsize=(7, 7))
    plt.scatter(x_9d[:, 0], x_9d[:, 2], c=label_color, alpha=0.5)
    plt.show()

    return X_clustered

