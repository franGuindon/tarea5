# -*- coding: utf-8 -*-
"""
@author: Francis, Alejandro
"""

print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"Gotta make a printing function"
def write_down(estimator="",name="",data=""):
    col_labels = ["Initializer", "",
                  "Time (s)", ".3f",
                  "Inertia", ".0f",
                  "Homogeneity", ".3f",
                  "Completeness", ".3f",
                  "V-Measure", ".3f",
                  "ARI", ".3f",
                  "AMI", ".3f",
                  "Silhouette", ".3f"]
    cell_width = 13
    columns = len(col_labels)//2
    def hline():
        print(cell_width*columns*'_')
    if name == "header":
        hline()
        print("\nWelcome!\n\n"\
              "This code will train several k-means models under varying conditions.\n"\
              "Metrics for each model will be sumarized in the following tables.\n"\
              "k-means++ initialized model centroids will be plotted as images.\n"\
              "Data and centroids for these same models will be reduced to 2 dimensions for visualization.\n")
        hline()
        print("\nFor the sklearn digits dataset:\n\n"\
              "Number of digits: {}\n"\
              "Number of samples: {}\n"\
              "Number of features: {}"\
              .format(n_digits,n_samples,n_features))
        hline()
    elif name[0] == "_":
        print("\n"+name[1:])
        hline()
        print("".join("{:{width}}".format(label,width=cell_width)\
                      for i,label in enumerate(col_labels) if i%2==0)
              )
    elif name == "footer":
        hline()
    else:
        t0 = time()
        estimator.fit(data)
        print("".join("{:<"+str(cell_width)+form+"}"\
                      for i,form in enumerate(col_labels) if i%2==1)\
               .format(name,
                       time()-t0,
                       estimator.inertia_,
                       metrics.homogeneity_score(labels,estimator.labels_),
                       metrics.completeness_score(labels,estimator.labels_),
                       metrics.v_measure_score(labels, estimator.labels_),
                       metrics.adjusted_rand_score(labels, estimator.labels_),
                       metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                       metrics.silhouette_score(data, estimator.labels_,
                                                metric='euclidean',
                                                sample_size=sample_size)
                       )
              )

#%%
"Global Variables"
np.random.seed(42)

scaler = StandardScaler()
X_digits, y_digits = load_digits(return_X_y=True)
data = scaler.fit_transform(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits # Used in metrics

sample_size = 300 # Used for silhouette metric
wholedatapca = PCA(n_components = 2)
centroidpca = PCA(n_components = 2)
reduced_data = wholedatapca.fit_transform(data)

# Grid points
h = .02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]
# high_grid_points = wholedatapca.inverse_transform(grid_points)
#%%
"Main Loop"
write_down(name = "header") # Introduction
for k in [3,10,20]:
    
    "Save k-means++ model"
    model = KMeans(init="k-means++",n_clusters=k,n_init=10)
    
    "Log metrics"
    write_down(name = "_For {} Clusters".format(k)) # Section title
    write_down(model,"k-means++",data) # Log k-means ++ model
    write_down(KMeans(init="random",n_clusters=k,n_init=10),"random",data) # Log random model
    write_down(KMeans(init=PCA(n_components=k).fit(data).components_, n_clusters=k, n_init=1),
               "PCA-based",data) # Log PCA based model
    write_down(name = "footer") # Section separator
    
    "Centroid images (special thanks to Mr Data Science at Medium.com)"
    centroids = model.cluster_centers_
    denormed_centroids = scaler.inverse_transform(centroids)
    num_row = 2**int(k/10) # 1 if k=3, 2 if k=10, 4 if k=20
    num_col = 5 if k != 3 else 3 # 3 if k=3, 5 otherwise
    fig, axes = plt.subplots(num_row, num_col, figsize = (2*num_col,2.5*num_row)) # Subplot grid
    for i in range(k):
        ax = axes[i//num_col, i%num_col] if k != 3 else axes[i] # Subplot coordinates
        ax.imshow(denormed_centroids[i].reshape(8,8), cmap="gray")
        ax.set_title("Centroid #{}".format(i+1))
    plt.tight_layout()
    plt.show()
    
    "Scatter and partition plots (data PCA)"
    reduced_model = KMeans(init="k-means++",n_clusters=k,n_init=10)
    reduced_model.fit(reduced_data)
    reduced_centroids = reduced_model.cluster_centers_
    points_pred = reduced_model.predict(grid_points).reshape(xx.shape)
    #points_pred = reduced_model.predict(high_grid_points).reshape(xx.shape)
    plt.figure(k)
    plt.clf()
    plt.imshow(points_pred, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.tab20,
               aspect='auto', origin='lower')
    # 2D Data Scatter
    batch_size = 500
    print_data = np.array(random.sample(reduced_data.tolist(),batch_size))
    plt.plot(print_data[:, 0], print_data[:, 1], 'k.', markersize=2)
    # 2D Centroid Scatter
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1],
            marker='p', s=70, linewidths=2,
            color='w', zorder=10)
    # 64D Centroid Scatter
    whole_centroids = wholedatapca.transform(centroids)
    plt.scatter(whole_centroids[:, 0], whole_centroids[:, 1],
            marker='*', s=70, linewidths=2,
            color='b', zorder=10)
    # Figure formatting
    plt.title("DIGIT CLUSTERING\n{} clusters, whole data PCA-reduction".format(k))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
    # IOU: This code block is quite similar to previous one and in the future
    #      should be modulated
    "Scatter and partition plots (centroid PCA)"
    # fit PCA with model centroids, then transform centroids and data
    centroid_centroids = centroidpca.fit_transform(centroids)
    centroid_reduced_data = centroidpca.transform(data)
    # train new reduced model with centroid transformed data
    centroid_reduced_model = KMeans(init="k-means++",n_clusters=k,n_init=10)
    centroid_reduced_model.fit(centroid_reduced_data)
    centroid_reduced_centroids = centroid_reduced_model.cluster_centers_
    points_pred = centroid_reduced_model.predict(grid_points).reshape(xx.shape)
    #points_pred = centroid_model.predict(high_grid_points).reshape(xx.shape)
    plt.figure(k)
    plt.clf()
    plt.imshow(points_pred, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.tab20,
               aspect='auto', origin='lower')
    # 2D Data Scatter
    batch_size = 500
    print_data = np.array(random.sample(centroid_reduced_data.tolist(),batch_size))
    plt.plot(print_data[:, 0], print_data[:, 1], 'k.', markersize=2)
    # 2D Centroid Scatter
    plt.scatter(centroid_reduced_centroids[:, 0], centroid_reduced_centroids[:, 1],
            marker='p', s=70, linewidths=2,
            color='w', zorder=10)
    # 64D Centroid Scatter
    plt.scatter(centroid_centroids[:, 0], centroid_centroids[:, 1],
            marker='*', s=70, linewidths=2,
            color='b', zorder=10)
    # Figure formatting
    plt.title("DIGIT CLUSTERING\n{} clusters, centroid PCA-reduction".format(k))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
#%%
# Debug block

# model = KMeans(init="k-means++",n_clusters=k,n_init=10)
# model.fit(data)
# datum = [1,2]
# pca_obj = PCA(n_components = 2)
# new_datum = pca_obj.fit(data).inverse_transform(datum).reshape(1,-1)
# print(new_datum)
# print(model.predict(new_datum))

# A = np.arange(1,5,1)
# B = np.arange(2,4,1)
# X,Y = np.meshgrid(A,B)
# #print(np.c_[X.ravel(),Y.ravel()])

# grid = [[1,2],
#         [2,3],
#         [3,4]]
# print(random.sample(grid,2))