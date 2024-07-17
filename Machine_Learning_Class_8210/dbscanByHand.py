# Daisy Fry Brumit
# BINF 8210 Final Project
# DBSCAN Class Build

import numpy as np
import pandas as pd
import matplotlib.pyplot as plotLib
from scipy.sparse import data
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def main():
# read in data (or make my own sample data using code from SKLearn)
#     centers = [[1, 1], [-1, -1], [1, -1]]
#     X, labels_true = make_blobs(
#         n_samples=750, centers=centers, cluster_std=0.4, random_state=0
#     )
#     X = StandardScaler().fit_transform(X)

## plot k-means clustering result

## testing dataset ####################
    dataframe = load_digits().data
    pca = PCA(2)

    BaseException = pca.fit_transform(dataframe)

    X = pd.DataFrame(dataframe).to_numpy()
    #########################################

    print('dataset dimensions:',X.shape)
    dbscanObject = DBSCAN(10, 0.3)
    clusterOutput = dbscanObject.initialClusterAssignments(dataset=X)
    print("cluster count = ",max(clusterOutput),'\nCluster labels:\n',clusterOutput)
    dbscanObject.visualize(dataset= X, clusterList= clusterOutput)

class DBSCAN:
    def __init__(self, minSamples, eps):
        self.minSamples = minSamples
        self.eps = eps

    def getNeighbors(self, pointIndex, dataset): # returns a list of neighbors
        neighborIndices = []
        for altPointIndex, altPoint in enumerate(dataset):
            if sum((self.point - altPoint)**2)**0.5 <= self.eps:
                neighborIndices.append(altPointIndex)
        return neighborIndices

    def initialClusterAssignments(self,dataset): # returns a list of cluster labels per point
        clusterLabel = 1
        self.clusterList = [-1]*len(dataset) # start every point as -1 to signify unlableled
        for pointIndex, self.point in enumerate(dataset):
            if self.clusterList[pointIndex] != -1:
                continue # skip the point if it already has a cluster label
            if len(self.getNeighbors(pointIndex, dataset)) >= self.minSamples:
                # x is a core point
                self.clusterList[pointIndex] = clusterLabel
                # Visit neighbors. Set this as its own function so I can call it again for every core point found
                self.clusterExpansion(pointIndex, dataset, clusterLabel)
            clusterLabel += 1

        return self.clusterList      

    def clusterExpansion(self, pointIndex, dataset, clusterLabel): # returns nothing
        for neighborIndex in self.getNeighbors(pointIndex, dataset):
            if self.clusterList[neighborIndex] == -1 or self.clusterList[neighborIndex] == 0:
                if len(self.getNeighbors(pointIndex, dataset)) >= self.minSamples: # if this point qualifies as a core point
                    self.clusterList[neighborIndex] = clusterLabel # reassign to this cluster
                    self.clusterExpansion(neighborIndex,dataset,clusterLabel) # continue expanding
                elif len(self.getNeighbors(pointIndex, dataset)) < self.minSamples:
                    self.clusterList[neighborIndex] = 0

    def visualize(self, dataset, clusterList):
        colorOptions =  ['#59C9A5', '#D81E5B', '#FFFD98', '#B9E3C6', '#23395B']
        fig, ax = plotLib.subplots()
        for i in range(max(clusterList)):
            color = colorOptions[i % len(colorOptions)]

            x, y = [], []
            for j in range(len(dataset)):
                if clusterList[j] == i:
                    x.append(dataset[j,0])
                    y.append(dataset[j,1])
            ax.scatter(x, y, c=color)
            plotLib.title('Estimated No. of Clusters: 3')
        plotLib.show()        

if __name__ == '__main__':
    main()