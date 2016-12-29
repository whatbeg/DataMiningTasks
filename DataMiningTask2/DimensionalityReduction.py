#! usr/bin/python
# coding=utf-8
"""
File Name: DimensionalityReduction.py
Description: Implementation of PCA, SVD, ISOMAP dimensionality reduction Method
Date: 2016-10-1
Author: QIU HU
Student ID: MG1633031
Some Statement: Because of python reference type, when mat(matrix) and list 's some [i][j] was assigned,
                then all row will be assigned, but array or ndarray will not, so sometimes type casting is necessary!
"""

import time
import os
from sklearn import manifold, datasets
import networkx as nx
import numpy as np
from numpy import linalg
import warnings
import math
import sys
import codecs
import traceback
reload(sys)
sys.setdefaultencoding('utf-8')
# warnings.filterwarnings("ignore")
infinite = float(1 << 53)


def handle_error():
    print("THERE IS AN ERROR")
    print(traceback.format_exc())


def dfs(i, graph, vis, color, c):

    if vis[i]:
        return
    vis[i] = 1
    # print("color[{}].append({})".format(c, i))
    color[c].append(i)
    for j, num in enumerate(graph[i]):
        # print(i, j, graph[i][j], )
        if i != j and num < infinite:
            dfs(j, graph, vis, color, c)
    return


class DimensionalityReduction(object):

    def __init__(self, iso, train_path, test_path):

        self.train = train_path
        self.test = test_path
        self.iso = iso
        self.datamat = np.mat([])
        self.dataset = []
        self.label = []

    def get_data(self, file):

        with open(file, 'r') as f:
            for line in f.readlines():
                x = line.strip().split(',')
                self.label.append(int(x[-1]))  # '1' or '-1'
                self.dataset.append(map(float, x[:-1]))  # for example, turn string '0.22302' to float 0.22302

    def load_dataset(self):

        self.get_data(self.train)
        if self.iso:
            self.get_data(self.test)
        self.datamat = np.mat(self.dataset)
        # print(shape(self.datamat))

    def pca(self, k):         # reduce to k-dimension

        datamat = self.datamat
        m = datamat.shape[0]                               # data item num
        datamat = datamat - datamat.mean(axis=0)           # 1. centralization
        cov = (datamat.T * datamat) * 1.0 / m              # 2. covariance matrix
        eigenval, eigenvector = linalg.eig(np.array(cov))  # 3. eigenvalue and eigenvector
        eigenval_index = np.argsort(eigenval)
        largest_k_eigenvector_index = eigenval_index[:-k-1:-1]
        W = []
        eigen_sum = 0.0
        for ind in largest_k_eigenvector_index:
            eigen_sum += eigenval[ind]
            W.append(map(float, list(eigenvector[:, ind])))
        return np.mat(W).T, eigen_sum/sum(eigenval)    # for k=10, return (60 * 10) W = (w1, w2, ..., wk), W.T * xi(d-dimension) -> yi(k-dimension)

    def svd(self, k):

        datamat = self.datamat.T
        u, sigma, vt = linalg.svd(datamat)
        return u[:, :k], float(sum(sigma[:k])) / sum(sigma)  # (60 * 10) W = (u1, u2, ..., uk), W.T * xi(d-dimension) -> yi(k-dimension)

    def mds(self, graph, k):

        try:
            n = graph.shape[0]
            sq_graph = np.array(graph) ** 2
            sq_dist_i = np.array([0.0 for _ in range(n)])
            sq_dist_j = np.array([0.0 for _ in range(n)])
            sq_dist = float(np.sum(sq_graph)) / (n ** 2)
            for i in range(n):
                sq_dist_i[i] = float(np.sum(sq_graph[i])) / n
            for j in range(n):
                sq_dist_j[j] = float(np.sum(sq_graph[:, j])) / n
            B = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    # print(sq_graph[i][j] , sq_dist_i[i] , sq_dist_j[j] , sq_dist)
                    B[i][j] = -0.5 * (sq_graph[i][j] - sq_dist_i[i] - sq_dist_j[j] + sq_dist)
                    B[j][i] = B[i][j]
            # print(B)
            eigenval, eigenvector = linalg.eigh(B)  # 3. eigenvalue and eigenvector
            eigenval_index = np.argsort(eigenval)
            largest_k_eigenvector_index = eigenval_index[:-k-1:-1]
            V = []
            eigen_sum = 0.0
            # print(eigenval[largest_k_eigenvector_index])
            for ind in largest_k_eigenvector_index:
                eigen_sum += abs(eigenval[ind])
                V.append(map(float, list(eigenvector[:, ind])))
            return np.diag(map(math.sqrt, eigenval[largest_k_eigenvector_index])) * np.mat(V), eigen_sum / (1 if sum(map(abs, eigenval)) == 0 else sum(map(abs, eigenval)))
        except:
            handle_error()

    def knn(self, inX, trainmat, near_neighbor):

        data_num = trainmat.shape[0]
        diff_mat = np.array(np.tile(inX, (data_num, 1)) - trainmat) ** 2  # (x1-y1)^2, ... , (xd'-yd')^2
        distance = np.array(np.sum(diff_mat, axis=1)) ** 0.5
        dis_sort_index = np.argsort(distance)
        return dis_sort_index[:near_neighbor], trainmat[dis_sort_index[:near_neighbor], :]

    def euclidean_dis(self, X, Y):

        sq_dis = np.array(X - Y) ** 2
        dis = np.sum(sq_dis) ** 0.5
        return dis

    def store_graph(self, graph):   # 存储图的上三角矩阵

        with codecs.open('graph.txt', 'w', 'utf-8') as f:
            for i in range(graph.shape[0]):
                for j in range(i, graph.shape[1]):
                    f.write("%.7f," % graph[i][j])
                f.write('\n')

    def read_floyd_graph(self, graph, filepath = 'graph_floyded.txt'):  # 读出上三角矩阵

        with codecs.open(filepath, 'r', 'utf-8') as f:
            i = 0
            for line in f.readlines():
                lis = map(float, line.strip().split(',')[:-1])
                graph[i][i:] = np.array(lis)
                i += 1
        return graph

    def connected_component(self, graph):

        n = graph.shape[0]
        vis = [0 for _ in range(n)]
        color = [[] for _ in range(n)]
        c = 0
        for i in range(n):
            if not vis[i]:
                # print("HIT: {}".format(i))
                dfs(i, graph, vis, color, c)
                # print(sorted(color[c]))
                c += 1
        # c is number of color
        print("Component: {}".format(c))
        # print(color[0])
        # print(sorted(color[1]))
        while c > 1:
            t_c, t_i, t_j, t_dis = 1, 0, 0, infinite
            for i in color[0]:
                for cl in range(1, c):
                    for j in color[cl]:
                        dis = self.euclidean_dis(self.datamat[i], self.datamat[j])
                        if dis < t_dis:
                            t_dis = dis
                            t_i = i
                            t_j = j
                            t_c = cl
            graph[t_i][t_j] = graph[t_j][t_i] = t_dis  # connect two closest group
            print("graph[{}][{}] = {}".format(t_i, t_j, t_dis))
            color[0].extend(color[t_c])
            for i in range(t_c, c-1):
                color[i] = color[i+1]
            c -= 1
        return graph

    def isomap(self, k, near_neighbor, first):

        data_size = self.datamat.shape[0]
        graph = np.array([[infinite] * data_size] * data_size)  # if type of graph is list, then below graph[i][j] = xx
        for i in range(self.datamat.shape[0]):               # will put ith row all xx, so, convert it to array type.
            index, near_x = self.knn(self.datamat[i], self.datamat, near_neighbor)
            for j in index:
                graph[i][j] = graph[j][i] = self.euclidean_dis(self.datamat[i], self.datamat[j])

        # <--neighbor connection graph constructed-->
        # print(graph)
        # print("Graph Construction Finished!")
        if first:
            graph = self.connected_component(graph)  # 使连通分量联通
            self.store_graph(graph)        # graph persistence because of long time of running floyd to get graph
            os.system(".\\floyd.exe")      # execute c++ program to run floyd progress
        graph = self.read_floyd_graph(graph)  # keep size constant
        # print(graph)
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if i > j:
                    graph[i][j] = graph[j][i]
                if graph[i][j] >= infinite-1:
                    graph[i][j] = 0
                    print("HAHAHAHAHAH")
        # print("Computation end!")
        # print(graph)
        return self.mds(graph, k)

    def sklearn_isomap(self, k, near_neighbor):

        return manifold.Isomap(near_neighbor, k).fit_transform(self.datamat).T, 0.0

if __name__ == '__main__':

    t = time.time()
    dim_red = DimensionalityReduction(True, 'sonar-train.txt', 'sonar-test.txt')
    # dim_red = DimensionalityReduction(True, 'splice-train.txt', 'splice-test.txt')
    dim_red.load_dataset()
    res, ratio = dim_red.isomap(10, 4, True)
    # Y = manifold.Isomap(6, 10).fit_transform(dim_red.datamat)
    # print(Y)
    # print(res.shape, ratio)
    # print(res)
    time_cost = time.time() - t
    print("%.1f" % time_cost)
    # dim_red2 = DimensionalityReduction(False, 'splice-train.txt', 'splice-test.txt')
    # dim_red2.load_dataset()



