#! usr/bin/python
# coding=utf-8
"""
File Name: DimensionalityReduction.py
Description: Test of PCA, SVD, ISOMAP dimensionality reduction Method
Date: 2016-10-1
Author: QIU HU
Student ID: MG1633031
"""

import time
import DimensionalityReduction as dim_red
from numpy import *
from operator import itemgetter
import collections
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.setrecursionlimit(1000000)  # 递归深度设置为1000000，以防dfs爆栈


class Test(object):

    def __init__(self, train_path, test_path):

        self.train = train_path
        self.test = test_path
        self.W = mat([])
        self.trainmat = []
        self.testmat = []
        self.rightlabel = []
        self.trainlabel = []

    def load_trainset(self):
        with open(self.train, 'r') as f:
            for line in f.readlines():
                x = line.strip().split(',')
                self.trainmat.append(map(float, x[:-1]))
                self.trainlabel.append(int(x[-1]))
        self.trainmat = mat(self.trainmat).T

    def load_testset(self):
        with open(self.test, 'r') as f:
            for line in f.readlines():
                x = line.strip().split(',')
                self.testmat.append(map(float, x[:-1]))
                self.rightlabel.append(int(x[-1]))
        self.testmat = mat(self.testmat).T   # testmat -> testmat.T, every feature list as a column, xi = [xi1, xi2, .., xid].T

    def project(self, testortrainmat):   # project the original d-dimension matrix to k-dimension matrix

        Y = self.W.T * testortrainmat
        return Y.T  # for example, k = 10, then Y.T is shape(103, 10)

    def knn(self, inX, trainmat, k):

        data_num = trainmat.shape[0]
        diff_mat = array(tile(inX, (data_num, 1)) - trainmat) ** 2  # (x1-y1)^2, ... , (xd'-yd')^2
        distance = array(sum(diff_mat, axis=1)) ** 0.5
        dis_sort_index = argsort(distance)
        classcount = collections.defaultdict(int)
        for i in range(k):
            votelabel = self.trainlabel[dis_sort_index[i]]
            classcount[votelabel] += 1
        sortedclass = sorted(classcount.iteritems(), key=itemgetter(1), reverse=True)
        return sortedclass[0][0]  # return class with biggest vote num

    def test_framwork(self, func, k):  # PCA, SVD的测试框架
        t = time.time()
        self.W, eigen_ratio = func(k)  # return a project matrix W
        self.load_trainset()
        self.load_testset()
        proj_train = self.project(self.trainmat)  # projected train set
        proj_test = self.project(self.testmat)    # projected test set
        i = 0
        right = 0
        for inX in proj_test:
            predictlabel = self.knn(inX, proj_train, 1)  # 1-NN
            if predictlabel == self.rightlabel[i]:
                right += 1
            i += 1
        time_cost = time.time() - t
        print("k = %d, eigen_ratio: %.2f%% Total test data: %d, right: %d, Accuracy: %.2f%%, Time: %.1f"
        % (k, 100*eigen_ratio, proj_test.shape[0], right, 100 * float(right)/proj_test.shape[0], time_cost))

    def testpca(self, k):

        dimension_reduction = dim_red.DimensionalityReduction(False, self.train, self.test)
        dimension_reduction.load_dataset()
        self.test_framwork(dimension_reduction.pca, k)

    def testsvd(self, k):

        dimension_reduction = dim_red.DimensionalityReduction(False, self.train, self.test)
        dimension_reduction.load_dataset()
        self.test_framwork(dimension_reduction.svd, k)

    def testisomap(self, trainsize, k, neigbor, first):
        t = time.time()
        dimension_reduction = dim_red.DimensionalityReduction(True, self.train, self.test)  # use both 2 dataset to train
        dimension_reduction.load_dataset()
        self.W, eigen_ratio = dimension_reduction.isomap(k, neigbor, first)  # return a projected sample set (10 * 208) Z(W) = [z1, z2, ..., zn]
        self.load_trainset()
        self.load_testset()
        proj_train = self.W[:, :trainsize].T  # projected train set  (trainsize, k)
        proj_test = self.W[:, trainsize:].T   # projected test set   (testsize, k)
        # print(proj_train.shape, proj_test.shape)
        i = 0
        right = 0
        for inX in proj_test:
            predictlabel = self.knn(inX, proj_train, 1)  # 1-NN
            if predictlabel == self.rightlabel[i]:
                right += 1
            i += 1
        time_cost = time.time() - t
        print("k = %d, eigen_ratio: %.2f%% Total test data: %d, right: %d, Accuracy: %.2f%%, Time: %.1f"
        % (k, 100 * eigen_ratio, proj_test.shape[0], right, 100 * float(right) / proj_test.shape[0], time_cost))


if __name__ == '__main__':

    k_set = [10, 20, 30]
    print("PCA Dimensionality Reduction")
    print("For Sonar Data Set (105, 60), (103, 60):")
    for k in k_set:
        test_sonar = Test('sonar-train.txt', 'sonar-test.txt')
        test_sonar.testpca(k)

    print("--------------------------------------------------------------------------------")
    print("For Splice Data Set (1000, 60), (2175, 60):")
    for k in k_set:
        test_splice = Test('splice-train.txt', 'splice-test.txt')
        test_splice.testpca(k)

    print("\nSVD Dimensionality Reduction")
    print("For Sonar Data Set:")
    for k in k_set:
        test_sonar = Test('sonar-train.txt', 'sonar-test.txt')
        test_sonar.testsvd(k)
    print("--------------------------------------------------------------------------------")
    print("For Splice Data Set:")
    for k in k_set:
        test_splice = Test('splice-train.txt', 'splice-test.txt')
        test_splice.testsvd(k)

    print("\nISOMAP Dimensionality Reduction")
    print("For Sonar Data Set:")
    for k in k_set:
        test_sonar = Test('sonar-train.txt', 'sonar-test.txt')
        test_sonar.testisomap(105, k, 4, first=(True if k == 10 else False))
    print("--------------------------------------------------------------------------------")
    print("For Splice Data Set:")
    for k in k_set:
        test_splice = Test('splice-train.txt', 'splice-test.txt')
        test_splice.testisomap(1000, k, 5, first=(True if k == 10 else False))

