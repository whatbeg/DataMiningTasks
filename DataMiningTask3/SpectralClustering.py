#! usr/bin/python
# coding=utf-8
"""
File Name: SpectralClustering.py
Description: Implementation of Spectral Clustering
Date: 2016-10-31
Author: QIU HU
"""

import numpy as np
import Tools
import kmedroids
from numpy.linalg import eigh


class SpectralClustering(object):

    def __init__(self, data, k, neighbors):

        self.dataSet, self.label, self.groundtruth = Tools.read_dataset(data)
        self.k = k
        self.neighbors = neighbors

    def construct_graph(self, dataSet, neighbors):

        n = self.dataSet.shape[0]   # number of data points
        dis_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dis_matrix[i][j] = dis_matrix[j][i] = Tools.Distance(dataSet, i, j)
        Ws = [np.zeros((n, n)) for _ in neighbors]
        for i in range(n):                                         # 10000
            smallest_dis_index = np.argsort(dis_matrix[i])
            for _, neb in enumerate(neighbors):                    # 3
                nearest_neighbor = smallest_dis_index[:neb+1]
                for nb in nearest_neighbor[1:]:                    # 9
                    Ws[_][i][nb] = 1
                    Ws[_][nb][i] = 1
        del dis_matrix
        # print("Nearest Neighborhood Graph Constructed~")
        Ds = [Ws[i] * -1 for i in range(len(Ws))]
        for i, W in enumerate(Ws):
            sigma_W = np.sum(W, axis=0)
            Ds[i] += np.diag(sigma_W)
        return Ds   # Laplacian Matrix

    def eigen_decomposition(self, Laplacian, k):

        eigenvalue, eigenvector = eigh(Laplacian)
        eigenval_index = np.argsort(eigenvalue)
        smallest_eig_index = eigenval_index[:k]
        V = []
        for eigv in smallest_eig_index:
            V.append(eigenvector[:, eigv])
        V = np.array(V).T    # (1000, 2)
        print(V.shape)
        return V

    def work(self, filename):

        Laplacians = self.construct_graph(self.dataSet, self.neighbors)   # Cost most of time !!!
        for i in range(len(Laplacians)):
            new_datamat = self.eigen_decomposition(Laplacians[i], self.k)
            kmedroids.Test(filename, new_datamat, self.label, self.groundtruth, self.k, self.neighbors[i])

if __name__ == '__main__':

    print("german dataSet")
    spectral = SpectralClustering('german.txt', 2, [3, 6, 9])
    spectral.work('german.txt')

    print("mnist dataSet")
    spectral = SpectralClustering('mnist.txt', 10, [3, 6, 9])
    spectral.work('mnist.txt')

