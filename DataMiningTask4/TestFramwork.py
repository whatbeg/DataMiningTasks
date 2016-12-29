#! usr/bin/python
# coding=utf-8
"""
File Name: TestFramework.py
Description: Test Result w of two algorithms
Date: 2016-11-06
Author: QIU HU
"""

import LogisticRegression as lr
import numpy as np
import Tools


class TestFramework(object):

    def __init__(self, traindata, testdata):

        self.testset, self.labels = self.load_testset(testdata)
        self.traindata = traindata

    def load_testset(self, data):

        datamat = np.loadtxt(data, delimiter=',')
        bias = np.ones(datamat.shape[0])  # bias add as making xi -> (1, xi)
        dataset = datamat[:, :-1]
        ndataset = np.insert(dataset, 0, values=bias, axis=1)
        print("Test dataset: {}".format(ndataset.shape))
        labels = datamat[:, -1]
        return ndataset, labels

    def test_logisticregression(self, display, train_or_test, w, testset, labels):

        err = 0
        m, d = testset.shape
        for i in range(m):
            predict = Tools.classify(w, testset[i])
            if predict != labels[i]:
                err += 1
        err_rate = float(err) / m
        if display:
            print("{} Error Rate: {:.4f}".format(train_or_test, err_rate))
        return err_rate

    def test_ridgeregression(self, display, train_or_test, w, testset, labels):

        err = 0
        m, d = testset.shape
        for i in range(m):
            predict = Tools.classify_ridge(w, testset[i])
            if predict != labels[i]:
                err += 1
        err_rate = float(err) / m
        if display:
            print("{} Error Rate: {:.4f}".format(train_or_test, err_rate))
        return err_rate

if __name__ == '__main__':

    test = TestFramework('dataset1-a9a-training.txt', 'dataset1-a9a-testing.txt')
