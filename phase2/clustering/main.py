import numpy as np
import pandas as pd  
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#from word_embedding.fasttext_data_loader import FastTextDataLoader
#from word_embedding.fasttext_model import FastText
from dimension_reduction import DimensionReduction
from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils
import time

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
# NOTE: I did it in word_embedding phase
# NOTE: for below line use clustering_train_data.csv path on google colab on google colab
data = pd.read_csv('clustering_train_data.csv')
X = np.array(data.drop('label', axis=1))
y = []
for i in range(len(X)) : 
    X[i] = np.array(X[i])
y = np.array(data['label']) 

# 1. Dimension Reduction
#     Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.
DR = DimensionReduction()

DR.wandb_plot_explained_variance_by_components(X, 'MIR', 'wandb_plot_explained_variance_by_components')

#     : Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
DR.wandb_plot_2d_tsne(X,'MIR', 'wandb_plot_2d_tsne')

new_X = DR.pca_reduce_dimension(X,2)

# 2. Clustering
## K-Means Clustering
# : Implement the K-means clustering algorithm from /.
# : Create document clusters using K-Means.
# : Run the algorithm with several different values of k.
#      For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# : Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# : Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
CU = ClusteringUtils()

for n_clusters in [2,5,7,10,12,15] :  
    start = time.time()
    CU.visualize_kmeans_clustering_wandb(new_X,n_clusters,'MIR', 'visualize_kmeans_clustering_wandb')
    end = time.time()
    print('time for k = ', n_clusters, 'is equal to : ', end-start)

CU.plot_kmeans_cluster_scores(new_X,y,[i for i in range(2,20)],'MIR','plot_kmeans_cluster_scores')
CU.visualize_elbow_method_wcss(new_X,[i for i in range(2,20)], 'MIR', 'visualize_elbow_method_wcss')
## Hierarchical Clustering
# : Perform hierarchical clustering with all different linkage methods.
# : Visualize the results.
CU.wandb_plot_hierarchical_clustering_dendrogram(new_X,'MIR', 'single' ,'wandb_plot_hierarchical_clustering_dendrogram-single')
CU.wandb_plot_hierarchical_clustering_dendrogram(new_X,'MIR', 'complete' ,'wandb_plot_hierarchical_clustering_dendrogram-complete')
CU.wandb_plot_hierarchical_clustering_dendrogram(new_X,'MIR', 'average' ,'wandb_plot_hierarchical_clustering_dendrogram-average')
CU.wandb_plot_hierarchical_clustering_dendrogram(new_X,'MIR', 'ward' ,'wandb_plot_hierarchical_clustering_dendrogram-ward')

# 3. Evaluation
# Using clustering metrics, evaluate how well your clustering method is performing.
n_clusters = 10
print()
print('cluster_kmeans')
centers, labels = CU.cluster_kmeans(new_X,n_clusters)
CM = ClusteringMetrics() 
print('silhouette_score : ', CM.silhouette_score(new_X,labels))
print('purity_score : ', CM.purity_score(y,labels))
print('adjusted_rand_score : ', CM.adjusted_rand_score(y,labels))

print()
print('cluster_hierarchical_single')
labels,_ = CU.cluster_hierarchical_single(new_X,n_clusters)
print('silhouette_score : ', CM.silhouette_score(new_X,labels))
print('purity_score : ', CM.purity_score(y,labels))
print('adjusted_rand_score : ', CM.adjusted_rand_score(y,labels))

print()
print('cluster_hierarchical_complete')
labels,_ = CU.cluster_hierarchical_complete(new_X,n_clusters)
print('silhouette_score : ', CM.silhouette_score(new_X,labels))
print('purity_score : ', CM.purity_score(y,labels))
print('adjusted_rand_score : ', CM.adjusted_rand_score(y,labels))

print()
print('cluster_hierarchical_average')
labels,_ = CU.cluster_hierarchical_average(new_X,n_clusters)
print('silhouette_score : ', CM.silhouette_score(new_X,labels))
print('purity_score : ', CM.purity_score(y,labels))
print('adjusted_rand_score : ', CM.adjusted_rand_score(y,labels))

print()
print('cluster_hierarchical_ward')
labels,_ = CU.cluster_hierarchical_ward(new_X,n_clusters)
CM = ClusteringMetrics() 
print('silhouette_score : ', CM.silhouette_score(new_X,labels))
print('purity_score : ', CM.purity_score(y,labels))
print('adjusted_rand_score : ', CM.adjusted_rand_score(y,labels))