#! usr/bin/python
# coding=utf-8
"""
File Name: LogisticRegression.py
Description: Logistic Regression Using SGD Tricks
Date: 2016-11-05
Author: QIU HU
"""
from Tools import *
import numpy as np
import random
import TestFramwork as tf
import matplotlib.pyplot as plt
import copy
import pylab as plt
import time


class LogisticRegression(object):

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
        dataset = self.normalization(dataset)   # Normaloization
        ndataset = np.insert(dataset, 0, values=bias, axis=1)
        print("Train dataset: {}".format(ndataset.shape))
        labels = datamat[:, -1]
        return ndataset, labels

    def finite_difference(self, w, dataset, labels):

        m, d = dataset.shape
        r = random.randint(0, m-1)
        z = dataset[r]
        print(np.dot(w, z))
        loss = np.log(1.0 + np.exp(-labels[r] * np.dot(w, z)))
        g = -labels[r] * z * sigmoid(-labels[r] * np.dot(w, z))
        gamma = 0.001
        tao = - gamma * g
        print("tao = {}".format(tao))
        w += tao
        loss_ = np.log(1.0 + np.exp(-labels[r] * np.dot(w, z)))
        print(loss_, loss, np.dot(tao, g))
        error = loss_ - loss - np.dot(tao, g)
        print(type(error))
        print("Finite Difference: {:.4f}".format(error))

    def calc_sparsity(self, w):

        return np.sum(w == 0.0)

    def StochasticGradientDescent(self, dataset, labels, iter, step):

        m, d = dataset.shape
        w = np.ones(d)    # w = [1, 1, 1, ..., 1]
        reg_lambda = 0.0001
        TF = tf.TestFramework(self.traindata, self.testdata)
        X = []
        TRAIN = []
        TEST = []
        OBJECT = []
        SPAR = []
        u = np.zeros(d)  # u: the abs value of the total L1-penalty that each weight could have received up to the point
        q = np.zeros(d)  # q: total L1-penalty that wi has actually received up to the point
        for _ in range(iter):
            print("Iteration {}...".format(_))
            chooseidx = list(range(m))
            fval = 0.0
            for i in range(m):
                alpha = 0.001
                u += alpha * reg_lambda
                r = random.randint(0, len(chooseidx)-1)
                idx = chooseidx[r]
                ybx = labels[idx] * np.dot(w, dataset[idx].T)
                w += alpha * (labels[idx] * sigmoid(-ybx) * dataset[idx] - reg_lambda * np.sign(w))
                # w += alpha * (labels[idx] * sigmoid(-ybx) * dataset[idx])  # without L1-regularization ( w ^ (1/2) )
                # wh = copy.deepcopy(w)
                # pos = w > 0.0
                # neg = w < 0.0
                # w[pos] -= (u[pos] + q[pos])
                # w[pos] = np.where(w[pos] < 0.0, 0.0, w[pos])
                # w[neg] += (u[neg] - q[neg])
                # w[neg] = np.where(w[neg] > 0.0, 0.0, w[neg])
                # q += w - wh
                fval += np.log(1.0 + np.exp(-ybx))   # Object Function
                del chooseidx[r]
            if _ % step == 0:
                train_err_rate = TF.test_logisticregression(0, "Train", w, dataset, labels)  # calc error in train set
                test_err_rate = TF.test_logisticregression(0, "Test", w, TF.testset, TF.labels)
                X.append(_ * m)
                TRAIN.append(train_err_rate)
                TEST.append(test_err_rate)
                fval = fval / m + reg_lambda * np.sum(np.abs(w))
                OBJECT.append(fval)
                sparsity = self.calc_sparsity(w)
                SPAR.append(float(sparsity) / d)

        # print("Final weights: {}".format(w))
        sparsity = self.calc_sparsity(w)
        print("Sparsity: {} / {}".format(sparsity, d))
        TF.test_logisticregression(1, "Train", w, dataset, labels)
        TF.test_logisticregression(1, "Test", w, TF.testset, TF.labels)

        plt.ylim(0, 1)
        plt.xlabel("Iteration times")
        plt.ylabel("Values")
        plt.plot(X, TRAIN, linestyle='-', marker='x', linewidth=1.5, label="Train Error Rate")
        plt.plot(X, TEST, linestyle='--', color='r', linewidth=1.5, label="Test Error Rate")
        plt.plot(X, OBJECT, linestyle='-.', linewidth=1.5, label="Object Function")
        plt.plot(X, SPAR, linestyle='-', linewidth=1.5, label="Sparsity")
        plt.legend(loc='upper right')
        plt.show()

    def work(self):

        self.StochasticGradientDescent(self.dataset, self.labels, iter=self.iter, step=int(self.iter*0.05))

    def scikit_logisticregression(self, dataset, labels):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty='l1', C=3.0, random_state=0, max_iter=100, n_jobs=2)
        lr.fit(dataset, labels)
        testset, truelabels = self.load_dataset(self.testdata)
        prob = lr.predict(testset)
        ans = prob * truelabels
        err_rate = float(np.sum(ans==-1)) / ans.shape[0]
        print("Scikit-Learn LR Test Error Rate: {:.4f}".format(float(err_rate)))

if __name__ == '__main__':

    # start = time.time()
    # LR = LogisticRegression('dataset1-a9a-training.txt', 'dataset1-a9a-testing.txt', 200)
    # LR.work()
    # end = time.time()
    # print("First Time Cost: {}".format(end-start))
    # del LR

    start = time.time()
    LR2 = LogisticRegression('covtype-training.txt', 'covtype-testing.txt', 30)
    LR2.work()
    end = time.time()
    print("Second Time Cost: {}".format(end-start))

    # LR.scikit_logisticregression(LR.dataset, LR.labels)
