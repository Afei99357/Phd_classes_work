# Author: Eric Liao /Daisy Brumit
# Date: November 1st, 2021

import argparse
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def kmeans(dataframe, k_clusters, num_of_iter):
    # step 1: get random sample ids as random centorids
    random_sample_ids = np.random.choice(len(dataframe), k_clusters, replace=False)
    centroids = dataframe.iloc[random_sample_ids, :]

    # step 2: find euclidean distances between centroids and all the sample points
    distances = cdist(dataframe, centroids, 'euclidean')

    # step 3: assign all data points according to the minimum distance
    points_assign_to_cluters = np.array([np.argmin(i) for i in distances]) # np.argmin() wii return the indices of the minimum values along an axis.

    for iteration in range(num_of_iter):
        new_centroids = []
        # for each cluster, update the centroids by find the mean of all the points of each cluster from previous clustring
        for i in range(k_clusters):
            update_centroids = dataframe.iloc[points_assign_to_cluters == i].mean(axis=0)
            new_centroids.append(update_centroids)

        # repeat step 1 to get the new centroids:
        new_centroids = np.vstack(new_centroids)

        # repeat step 2:
        new_distances = cdist(dataframe, new_centroids, 'euclidean')

        # repeat step 3:
        new_points_assign_to_clusters = np.array([np.argmin(i) for i in new_distances])

    return new_points_assign_to_clusters


def main():
    # parser = argparse.ArgumentParser(description="Reads factors for each sample and writes them into CSV file")
    #
    # # ### total 5 parameters need to specify, first 2 are required, last 3 are optional
    # parser.add_argument('--input_file', help='files contains matrix, either csv/txt, separate with spaces',
    #                     required=True)
    # parser.add_argument('--axis', nargs='?', const=1, type=int, default=0,
    #                     help='axis value will be integer number(0 or 1), default is 0, '
    #                          'if axis = 0, the structure of dataset is rows as samples, columns as features/variables; '
    #                          'if axis = 1, the structure of dataset is rows as features/variables, columns as samples.'
    #                          'The example of dataset with axis=0 and axis=1 is in the folder called data_file_example')
    # parser.add_argument('--header', nargs='?', const=1, type=int, default=0,
    #                     help='header value will be integer number(0 or 1), default is 0'
    #                          'if header = 0, the first row is not column name; '
    #                          'if header = 1, the first row is column name.')
    # parser.add_argument('--index', nargs='?', const=1, type=int, default=0,
    #                     help='index value will be integer number(0 or 1), default is 0'
    #                          'if index = 0, the first column is not index column; '
    #                          'if index = 1, the first column is index column.')
    #
    # args = parser.parse_args()
    #
    # input_file = args.input_file
    # axis = args.axis
    # header = args.header
    # index = args.index
    #
    # # check if the data table has column name and index column
    # if header == 0 and index == 0:
    #     dataframe = pd.read_csv(input_file)
    # if header == 1 and index == 1:
    #     dataframe = pd.read_csv(input_file, header=0, index_col=0)
    # if header == 0 and index == 1:
    #     dataframe = pd.read_csv(input_file, index_col=0)
    # if header == 1 and index == 0:
    #     dataframe = pd.read_csv(input_file, header=0)
    #
    # # original data table structure is row as features/variables and columns as samples
    # if axis == 1:
    #     dataframe = dataframe.T

    # dataframe = pd.read_csv('/Users/ericliao/PycharmProjects/Phd_Class/Machine_Learning_Class_8210/data_files/'
    #                         'iris_dataset.csv', index_col=0)
    # dataframe = dataframe.iloc[:, 0:4]

    # ## testing dataset ####################
    # dataframe = load_digits().data
    # pca = PCA(2)
    #
    # dataframe = pca.fit_transform(dataframe)
    #
    # dataframe = pd.DataFrame(dataframe)
    # #########################################

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )
    dataframe = StandardScaler().fit_transform(X)
    dataframe = pd.DataFrame(dataframe)

    labels = kmeans(dataframe, 3, 100)

    ## plot k-means clustering result
    fig, ax = plt.subplots()
    u_labels = np.unique(labels)

    for i in u_labels:
        ax.scatter(dataframe.iloc[labels == i, 0], dataframe.iloc[labels == i, 1], label=i)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
