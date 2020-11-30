# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:10:15 2020

@author: Francis
"""

"NEEDED FOR EVERYTHING"
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

scaler = StandardScaler()
X_digits, y_digits = load_digits(return_X_y=True)
data = scaler.fit_transform(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

"Gotta make a printing function"
def write_down(estimator="",name="",data=""):
    col_labels = ["Initializer", "}",
                  "Time (s)", ".3f}",
                  "Inertia", ".0f}",
                  "Homogeneity", ".3f}",
                  "Completeness", ".3f}",
                  "V-Measure", ".3f}",
                  "ARI", ".3f}",
                  "AMI", ".3f}",
                  "Silhouette", ".3f}"]
    cell_width = 13
    columns = len(col_labels)//2
    def hline():
        print(cell_width*columns*'_')
    if name == "header":
        hline()
        print("\nWelcome!\n\n"\
              "(introduction)")
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
        print("".join("{:<"+str(cell_width)+form\
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

# Introduction
write_down(name = "header")
for k in [3,10,20]:
    # Save k-means++ model
    model = KMeans(init="k-means++",n_clusters=k,n_init=10)
    # Log metrics
    write_down(name = "_For {} Clusters".format(k))
    write_down(model,"k-means++",data)
    write_down(KMeans(init="random",n_clusters=k,n_init=10),"random",data)
    write_down(KMeans(init=PCA(n_components=k).fit(data).components_, n_clusters=k, n_init=1),
               "PCA-based",data)
    write_down(name = "footer")
    # Centroid images
    centroids = scaler.inverse_transform(model.cluster_centers_)
    # centroid = centroids[0].reshape(8,8)
    # plt.figure(k)
    # plt.clf()
    # plt.imshow(centroid,
    #            interpolation="nearest",
    #            cmap="gray")
    # plt.show
    num_row = 2**int(k/10)
    num_col = 5 if k != 3 else 3
    fig, axes = plt.subplots(num_row, num_col, figsize = (2*num_col,2.5*num_row))
    for i in range(k):
        ax = axes[i//num_col, i%num_col] if k != 3 else axes[i]
        ax.imshow(centroids[i].reshape(8,8), cmap="gray")
        ax.set_title("Centroid #{}".format(i+1))
    plt.tight_layout()
    plt.show()

# # #############################################################################
# # Visualize the results on PCA-reduced data

# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# # #############################################################################
# #

# est = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# est.fit(data)
# centroids = est.cluster_centers_
# img = data[0].reshape(8,8)
# img2 = centroids[3].reshape(8,8)
# fig = plt.figure
# plt.imshow(img2,cmap="gray")
# plt.show