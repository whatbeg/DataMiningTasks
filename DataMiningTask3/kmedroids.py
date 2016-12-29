#! usr/bin/python
# coding=utf-8
"""
File Name: kmedoids.py
Description: Implementation of K-medoids Clustering
Date: 2016-10-30
Author: QIU HU
"""

import numpy as np
import random
import Tools
import copy
MAX_INT = (1 << 50)


class Kmedoids(object):

    def __init__(self, dataSet, label, gt, k):

        self.dataSet = np.array([])
        self.label = np.array([])
        self.groundtruth = []
        self.dataSet, self.label, self.groundtruth = dataSet, label, gt
        self.k = k
        self.center = self.get_initial_center(k)
        self.belong = [0] * self.dataSet.shape[0]

    def get_initial_center(self, k):   # Randomly choose cluster center for initial value

        n = self.dataSet.shape[0]      # number of data point
        chosed = []
        index = list(range(n))
        for _ in range(k):
            ind = random.randint(0, len(index)-1)
            chosed.append(ind)
            index.pop(ind)
        return chosed

    def assign_to_cluster(self, centers):

        for i, center_idx in enumerate(centers):
            self.belong[center_idx] = i
            # print("self.belong[{}] = {}".format(center_idx, i))
        n = self.dataSet.shape[0]
        for i in range(n):       # for every data point, assign the closest cluster to it
            mindis = MAX_INT
            tag_c = 0
            for c in centers:    # c is center index
                dis = Tools.Distance(self.dataSet, i, c)
                if dis < mindis:
                    mindis = dis
                    tag_c = self.belong[c]
            self.belong[i] = tag_c

    def choose_best_ik(self, dataset, idxs):    # O(Nk^2)

        MIN_DIS = MAX_INT
        tag = 0
        for i in idxs:
            sigma_Distance = 0.0
            for j in idxs:
                if j != i:
                    sigma_Distance += Tools.Distance(dataset, i, j)
            if sigma_Distance < MIN_DIS:
                MIN_DIS = sigma_Distance
                tag = i
        return tag

    def get_cluster_from_belong(self):

        n = self.dataSet.shape[0]
        cluster = [[] for _ in range(self.k)]
        for i in range(n):
            cluster[self.belong[i]].append(i)  # cluster c have i-th data point
        # Tools.pl2d(cluster)
        return cluster

    def change_center(self):

        cluster = self.get_cluster_from_belong()
        # Tools.pl2d(cluster)
        new_center = []
        for i in range(self.k):
            idx = self.choose_best_ik(self.dataSet, cluster[i])
            new_center.append(idx)
        return new_center

    def getM(self, label, gt, cluster):
        # print("gt = {}".format(gt))
        kt = len(gt)                  # number of ground truth classes
        kd = self.k                   # number of algorithm determined classes
        mapper = {}
        for i, c_label in enumerate(gt):
            mapper[c_label] = i          # mapper[1] = 0, mapper[-1] = 1
        # print(mapper)
        M = [[0.0] * kd for _ in range(kt)]  # M matrix
        for j in range(kd):
            for dp in cluster[j]:
                true_class = mapper[label[dp]]
                M[true_class][j] += 1
        return np.array(M)

    def work(self, center):
        print(center)
        while True:
            self.assign_to_cluster(center)
            # print(self.belong)
            new_center = self.change_center()
            if set(new_center) == set(center):
                break
            center = copy.deepcopy(new_center)
        cluster = self.get_cluster_from_belong()
        M = self.getM(self.label, self.groundtruth, cluster)
        # Tools.pl2d(cluster)
        gini = Tools.calc_GINI(M)
        purity = Tools.calc_Purity(M)
        print("Gini index is {}, and Purity is {}".format(gini, purity))
        return gini, purity


def Test(filename, dataSet, label, gt, k, neighbor=0):

    mingini = float(MAX_INT)
    best_purity = 0.0
    for _ in range(10):
        kmedoids = Kmedoids(dataSet, label, gt, k)
        gini, purity = kmedoids.work(kmedoids.center)
        if gini < mingini:
            mingini = gini
        if purity > best_purity:
            best_purity = purity
        del kmedoids
    print("-----------------------------------------------------------------")
    print("For {}, Neighbor {}, Minimum Gini index is {}, Purity = {}\n".format(filename, neighbor, mingini, best_purity))


if __name__ == '__main__':

    dataSet, label, gt = Tools.read_dataset('german.txt')
    Test('german.txt', dataSet, label, gt, 2)

    dataSet, label, gt = Tools.read_dataset('mnist.txt')
    Test('mnist.txt', dataSet, label, gt, 10)
