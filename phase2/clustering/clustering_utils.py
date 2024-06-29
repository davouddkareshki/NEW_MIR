import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb
import matplotlib.pyplot as plt

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from clustering_metrics import ClusteringMetrics
from dimension_reduction import DimensionReduction


class ClusteringUtils:
    def __init__(self):
        pass

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        #kmean = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        #kmean.fit(emb_vecs) 
        centers = []
        for _ in range(n_clusters) : 
            point = emb_vecs[random.randint(0,len(emb_vecs)-1)] 
            centers.append(point) 

        N = np.array(emb_vecs).shape[0]
        M = np.array(emb_vecs).shape[1]
        labels = np.zeros(N)
        for _ in range(max_iter) :     
            clusters = {}
            for i,point in enumerate(emb_vecs):
                nearest_distance = float('inf')
                nearest_center = None 
                nearest_center_idx = None 
                for j,center_point in enumerate(centers) : 
                    dis = np.linalg.norm(center_point-point)
                    if dis < nearest_distance :
                        nearest_distance = dis 
                        nearest_center = center_point
                        nearest_center_idx = j
                if nearest_center_idx not in clusters.keys() : 
                    clusters[nearest_center_idx] = []
                clusters[nearest_center_idx].append(point)
                labels[i] = nearest_center_idx

            new_centers = []
            for center_idx in clusters.keys() : 
                center = centers[center_idx]
                avreage = np.zeros(M)
                num = 0
                for point in clusters[center_idx] : 
                    num += 1 
                    avreage += point 
                avreage /= num 
                new_centers.append(avreage) 
            centers = new_centers 

        return (centers, labels)

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        tf = {} 
        for text in documents : 
            tokens = text.split(' ')
            for token in tokens : 
                if token not in tf.keys() : 
                    tf[token] = 0
                tf[token] += 1
        arr_tf = [] 
        for token in tf.keys() : 
            arr_tf.append((tf[token],token))
        arr_tf = sorted(arr_tf,reverse=True)[0:top_n]
        for i,val in enumerate(arr_tf) : 
            arr_tf[i] = (val[1],val[0]) 
        return arr_tf

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        centers,labels = self.cluster_kmeans(emb_vecs, n_clusters)
        WCSS = 0
        for point in emb_vecs:
            nearest_distance = float('inf')
            nearest_center = None 
            for center_point in centers : 
                dis = np.linalg.norm(center_point-point)
                if dis < nearest_distance :
                    nearest_distance = dis 
                    nearest_center = center_point 
            WCSS += nearest_distance 
        return (centers,labels,WCSS)
    
    def plot_dendrogram(self, model):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

        dendrogram(linkage_matrix)
        pass 
    
    def cluster_hierarchical_single(self, emb_vecs: List, n_clusters: int) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        model = AgglomerativeClustering(n_clusters = n_clusters, linkage='single', compute_distances = True)
        model.fit(emb_vecs)
        return model.labels_, model

    def cluster_hierarchical_complete(self, emb_vecs: List, n_clusters: int) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        model = AgglomerativeClustering(n_clusters = n_clusters, linkage='complete', compute_distances = True)
        model.fit(emb_vecs)
        return model.labels_, model

    def cluster_hierarchical_average(self, emb_vecs: List, n_clusters: int) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        model = AgglomerativeClustering(n_clusters = n_clusters, linkage='average', compute_distances = True)
        model.fit(emb_vecs)
        return model.labels_, model

    def cluster_hierarchical_ward(self, emb_vecs: List, n_clusters: int) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        model = AgglomerativeClustering(n_clusters = n_clusters, linkage='ward', compute_distances = True)
        model.fit(emb_vecs)
        return model.labels_, model

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        centers, labels = self.cluster_kmeans(data, n_clusters)
        DR = DimensionReduction()
        data = DR.convert_to_2d_tsne(data)
        #centers = DR.convert_to_2d_tsne(centers)

        # Plot the clusters
        plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis')

        # Log the plot to wandb
        wandb.log({'Kmean clustering... ':wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        run = wandb.init(project=project_name, name=run_name)

        # Perform hierarchical clustering
        model = None 
        if(linkage_method == 'single') :   _,model = self.cluster_hierarchical_single(data,n_clusters=10)
        if(linkage_method == 'complete') : _,model = self.cluster_hierarchical_complete(data,n_clusters=10)
        if(linkage_method == 'average') :  _,model = self.cluster_hierarchical_average(data,n_clusters=10)
        if(linkage_method == 'ward') :     _,model = self.cluster_hierarchical_ward(data,n_clusters=10)

        # Create dendrogram plot
        plt.figure(figsize=(10, 7))
        self.plot_dendrogram(model)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')

        # Log the plot to wandb
        wandb.log({"Hierarchical Clustering Dendrogram": wandb.Image(plt)})

        # Close the plot display window
        plt.close()

        # Create linkage matrix for dendrogram
        # TODO
    
        pass

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None, run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        dimension_K = [] 
        silhouette_scores = []
        purity_scores = []
        CM = ClusteringMetrics()
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            dimension_K.append(k)
            centers, labels = self.cluster_kmeans(embeddings,k)
            sil_score = CM.silhouette_score(embeddings,labels)
            pur_score = CM.purity_score(true_labels,labels)
            silhouette_scores.append(sil_score)
            purity_scores.append(pur_score)
        
        # Plotting the scores
        plt.plot(dimension_K, silhouette_scores, 'red')
        plt.plot(dimension_K, purity_scores, 'blue')

        # Logging the plot to wandb
        if project_name and run_name:
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Cluster Scores": wandb.Image(plt)})
        plt.close()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        dimension_K = []
        for k in k_values:
            _,_,wcc = self.cluster_kmeans_WCSS(embeddings, k) 
            wcss_values.append(wcc)
            dimension_K.append(k)

        # Plot the elbow method
        plt.plot(dimension_K, wcss_values)

        # Log the plot to wandb
        wandb.log({"Elbow Method": wandb.Image(plt)})

        plt.close()