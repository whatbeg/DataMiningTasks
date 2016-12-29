#! usr/bin/python
# coding=utf-8
"""
File Name: RidgeRegression.py
Description: Ridge Regression Using SGD Tricks
Date: 2016-11-06
Author: QIU HU
"""
from Tools import *
import numpy as np
import random
import TestFramwork as tf
# import matplotlib.pyplot as plt
import time
import pylab as plt


class RidgeRegression(object):

    def __init__(self, traindata, testdata, iter):

        self.dataset, self.labels = self.load_dataset(traindata)
        self.iter = iter
        self.traindata, self.testdata = traindata, testdata

    def normalization(self, dataset):

        maxi = np.max(dataset, axis=0)
        mini = np.min(dataset, axis=0)
        data = maxi - mini
        del maxi
        data = np.where(data <= 0, 1.0, data)
        dataset = (dataset - mini) / data
        return dataset

    def load_dataset(self, data):

        datamat = np.loadtxt(data, delimiter=',')
        bias = np.ones(datamat.shape[0])     # bias add as making xi -> (1, xi)
        dataset = datamat[:, :-1]
        dataset = self.normalization(dataset)  # Normaloization
        ndataset = np.insert(dataset, 0, values=bias, axis=1)
        print(ndataset.shape)
        labels = datamat[:, -1]
        return ndataset, labels

    def StochasticGradientDescent(self, dataset, labels, iter, step):

        m, d = dataset.shape
        w = np.ones(d)       # w = [1, 1, 1, ..., 1]
        reg_lambda = 0.0001  # Regularization Lambda
        TF = tf.TestFramework(self.traindata, self.testdata)
        X = []
        TRAIN = []
        TEST = []
        OBJECT = []
        print("step = {}".format(step))
        for _ in range(iter):
            print("Iteration {}...".format(_))
            chooseidx = list(range(m))
            fval = 0.0
            for i in range(m):
                alpha = 0.001 / (1.0 + 0.001*(_*m+i)*0.003)
                r = random.randint(0, len(chooseidx)-1)
                idx = chooseidx[r]
                error = labels[idx] - np.dot(w, dataset[idx])
                w += 2.0 * alpha * (error * dataset[idx] - reg_lambda * w)
                fval += error ** 2
                del chooseidx[r]
            if _ % step == 0:
                train_err_rate = TF.test_ridgeregression(0, "Train", w, dataset, labels)
                test_err_rate = TF.test_ridgeregression(1, "Test", w, TF.testset, TF.labels)
                X.append(_*m)
                TRAIN.append(train_err_rate)
                TEST.append(test_err_rate)
                fval = fval / m + reg_lambda * np.sum(w ** 2)
                OBJECT.append(fval)

        TF.test_ridgeregression(1, "Train", w, dataset, labels)
        TF.test_ridgeregression(1, "Test", w, TF.testset, TF.labels)

        plt.ylim(0, 1)
        plt.xlabel("Iteration times")
        plt.ylabel("Values")
        plt.plot(X, TRAIN, linestyle='-', marker='x', linewidth=1.5, label="Train Error Rate")
        plt.plot(X, TEST, linestyle='--', color='r', linewidth=1.5, label="Test Error Rate")
        plt.plot(X, OBJECT, linestyle='-.', linewidth=1.5, label="Object Function")
        plt.legend(loc='upper right')
        plt.show()

    def work(self):

        self.StochasticGradientDescent(self.dataset, self.labels, iter=self.iter, step=int(self.iter*0.05))

    def scikit_ridgeregression(self, dataset, labels):
        from sklearn.linear_model import RidgeClassifier
        lr = RidgeClassifier(fit_intercept=False, max_iter=100, random_state=0)
        lr.fit(dataset, labels)
        testset, truelabels = self.load_dataset(self.testdata)
        prob = lr.predict(testset)
        ans = prob * truelabels
        err_rate = float(np.sum(ans==-1)) / ans.shape[0]
        print("Scikit Learn RR Test Error Rate: {:.2f}".format(float(err_rate)))

if __name__ == '__main__':

    start = time.time()
    RR = RidgeRegression('dataset1-a9a-training.txt', 'dataset1-a9a-testing.txt', 20)
    RR.work()
    end = time.time()
    print("First Time Cost: {}".format(end - start))
    RR.scikit_ridgeregression(RR.dataset, RR.labels)
    del RR

    start = time.time()
    RR2 = RidgeRegression('covtype-training.txt', 'covtype-testing.txt', 20)
    RR2.work()
    end = time.time()
    print("Second Time Cost: {}".format(end - start))
    RR2.scikit_ridgeregression(RR2.dataset, RR2.labels)
