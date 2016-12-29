#! usr/bin/python
# coding=utf-8
"""
File Name: NaiveBayes.py
Description: Naive Bayes Classifier
Date: 2016-11-19 -> 27
Author: QIU HU
"""

import numpy as np
import collections
import math
from operator import itemgetter
import matplotlib.pylab as plt


def gauss(x, mu, sigma):
    if sigma == 0.0:
        print("ZERO SIGMA!!!")
        sigma = 1e-7
    return 1.0 / (math.sqrt(2 * math.pi) * sigma) * math.exp(-((x-mu)**2) / (2.0 * sigma**2))


def binary_search(rnd, Wsum):
    n = len(Wsum)
    low = 0
    high = n-1
    while low <= high:
        mid = (low + high) / 2
        if rnd > Wsum[mid]:
            low = mid + 1
        elif rnd < Wsum[mid]:
            high = mid - 1
        else:
            return mid
    return low


def pl_dic(dic, pl):
    print("{}:".format(pl))
    for key in dic:
        if isinstance(dic[key], dict):
            for t in dic[key]:
                if isinstance(dic[key][t], dict):
                    for s in dic[key][t]:
                        print("[{}][{}][{}] = {}".format(key, t, s, dic[key][t][s]))
                else:
                    print("[{}][{}] = {}".format(key, t, dic[key][t]))
        else:
            print("[{}] = {}".format(key, dic[key]))


class NaiveBayes(object):

    def __init__(self, data):

        self.dataset, self.catetype, self.labels, self.possible_vals = self.load_data(data)

    def load_data(self, data):

        dataset = np.loadtxt(data, delimiter=',')
        catetype = dataset[0, :-1]
        labels = dataset[1:, -1]
        possible_vals = {}
        for i in range(len(catetype)):
            if catetype[i] == 1.0:    # discrete
                possible_vals[i] = set(dataset[:, i])
        return dataset[1:, :-1], catetype, labels, possible_vals

    def gen_CV(self, dataset, labels, fold=10):

        m, d = dataset.shape
        random_idx = np.random.permutation(m)
        if m % fold > fold/2:
            batch_size = (m / fold + 1)        # 每份多少个
        else:
            batch_size = m / fold
        for i in range(fold):
            begin = batch_size*i
            if i == fold-1:
                end = m
            else:
                end = batch_size*(i+1)
            testset = dataset[random_idx[begin:end]]
            testlabels = labels[random_idx[begin:end]]

            trainset = dataset[np.append(random_idx[:begin], random_idx[end:])]
            trainlabels = labels[np.append(random_idx[:begin], random_idx[end:])]

            yield trainset, trainlabels, testset, testlabels

    def get_model(self, dataset, catetype, labels, possible_vals):
        # model[class][feature][param] = ...
        model = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
        class_prob = {}
        laplacian = 1.0
        m, d = dataset.shape
        class_num = len(set(labels))             # number of classes
        for cla in set(labels):                  # for every class
            index = np.where(labels == cla)[0]   # [1 2 5] (array) : index of data point of this class
            total = len(index)                   # total data samples of this class
            class_prob[cla] = float(total + laplacian) / (m + class_num*laplacian)
            for feature_idx in range(d):         # for every feature
                discrete = (catetype[feature_idx] == 1.0)
                if discrete:                     # if discrete feature
                    for val in possible_vals[feature_idx]:    # all conditions
                        cnt = len(np.where(dataset[index, feature_idx] == val)[0])   # cnt maybe 0
                        model[cla][feature_idx][val] = float(cnt + laplacian) / (total + len(possible_vals[feature_idx])*laplacian)
                        # print("model[{}][{}][{}] = ({} + 1) / ({} + {}*1".format(cla, feature_idx, val, cnt, total, len(possible_vals[feature_idx])))
                else:                            # if continuous feature
                    vals = dataset[index, feature_idx]
                    mu = np.mean(vals)
                    sigma = math.sqrt(np.var(vals))
                    model[cla][feature_idx]['mu'] = mu
                    model[cla][feature_idx]['sigma'] = sigma

        return model, class_prob

    def classify(self, x, model, class_prob, catetype):

        d = x.shape[0]
        probs = {}
        for cla in class_prob:                     # enumerate class
            prob = math.log(class_prob[cla])       # log(P(Y = ck))
            for feature_idx in range(d):           # enumerate every dimension
                val = x[feature_idx]
                if catetype[feature_idx] == 1.0:   # discrete
                    prob += math.log(model[cla][feature_idx][val])          # * P(Xj = x(j) | Y = ck)
                else:
                    # print(val, model[cla][feature_idx]['mu'], model[cla][feature_idx]['sigma'], gauss(val, model[cla][feature_idx]['mu'], model[cla][feature_idx]['sigma']))
                    prob += math.log(gauss(val, model[cla][feature_idx]['mu'], model[cla][feature_idx]['sigma']))
            probs[cla] = prob
        ans = sorted(probs.items(), key=itemgetter(1), reverse=True)
        return ans[0][0]

    def work(self):
        cv = 1
        accs = []
        for trainset, trainlabels, testset, testlabels in self.gen_CV(self.dataset, self.labels, 5):
            model, class_prob = self.get_model(trainset, self.catetype, trainlabels, self.possible_vals)
            m, d = testset.shape
            err = 0
            for i in range(m):
                predict = self.classify(testset[i], model, class_prob, self.catetype)
                if predict != testlabels[i]:
                    err += 1
            err_rate = float(err) / m
            accs.append(1.0 - err_rate)
            cv += 1
        mean = np.mean(accs)
        var = np.var(accs)
        print("[Naive Bayes Cross Validation] -> Mean: {:.4f}, Variance: {:.6f}".format(mean, var))
        return mean, var

    def sampling(self, W, dataset, labels):
        n_dataset = []
        n_labels = []
        Wsum = [W[0] for _ in range(len(W))]
        for i in range(1, len(W)):
            Wsum[i] = Wsum[i-1] + W[i]
        m, d = dataset.shape
        idxs = []
        for _ in range(m):     # sampling m data point
            rnd = np.random.uniform(0.0, 1.0)
            idx = binary_search(rnd, Wsum)
            idxs.append(idx)
            n_dataset.append(dataset[idx])
            n_labels.append(labels[idx])
        return np.array(n_dataset), np.array(n_labels)

    def get_err_rate(self, dataset, labels, model, class_prob, catetype):
        m, d = dataset.shape
        err = 0
        err_label = []
        for i in range(m):
            predict = self.classify(dataset[i], model, class_prob, catetype)
            if predict != labels[i]:
                err += 1
                err_label.append(1)
            else:
                err_label.append(0)
        err_rate = float(err) / m
        return err_label, err_rate

    def adaboost(self, dataset, labels, catetype, possible_vals, T):

        m, d = dataset.shape
        MODELS = []        # store classifiers
        CLASS_PROBS = []   # store classifiers
        ALPHA = []
        W = [1.0 / m for _ in range(m)]
        t = 0
        while True:
            t += 1
            n_dataset, n_labels = self.sampling(W, dataset, labels)    # 根据权重抽样
            model, class_prob = self.get_model(n_dataset, catetype, n_labels, possible_vals)
            err_label, err_rate = self.get_err_rate(n_dataset, n_labels, model, class_prob, catetype)
            if err_rate >= 0.5 or err_rate < 1e-7 or t > T:
                break
            alpha = 0.5 * math.log((1 - err_rate) / err_rate)
            MODELS.append(model)              # Store Model
            CLASS_PROBS.append(class_prob)
            ALPHA.append(alpha)
            for i in range(m):
                if err_label[i] == 1:         # misclassified instance
                    W[i] *= math.exp(alpha)
                else:
                    W[i] *= math.exp(-alpha)
            sumW = sum(W)
            for i in range(m):
                W[i] /= sumW
        return MODELS, CLASS_PROBS, ALPHA

    def get_err_rate_weight(self, dataset, labels, weight, model, class_prob, catetype):
        m, d = dataset.shape
        err = 0.0
        err_label = []
        for i in range(m):
            predict = self.classify(dataset[i], model, class_prob, catetype)
            if predict != labels[i]:
                err += weight[i]
                err_label.append(1)
            else:
                err_label.append(0)
        err_rate = float(err) / np.sum(weight)
        return err_label, err_rate

    def get_model_weight(self, dataset, catetype, labels, possible_vals, weight):
        model = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
        class_prob = {}
        m, d = dataset.shape
        laplacian = 1.0 / m
        sumWeight = np.sum(weight)
        class_num = len(set(labels))             # number of classes
        for cla in set(labels):                  # for every class
            index = np.where(labels == cla)[0]   # [1 2 5] (array) : index of data point of this class
            total = np.sum(weight[index])        # total data weights of this class
            class_prob[cla] = float(total + laplacian) / (sumWeight + class_num*laplacian)
            for feature_idx in range(d):         # for every feature
                discrete = (catetype[feature_idx] == 1.0)
                if discrete:                     # if discrete feature
                    for val in possible_vals[feature_idx]:    # all conditions
                        idx = np.where(dataset[index, feature_idx] == val)[0]   # 取该值的数据点的下标
                        cnt = np.sum(weight[idx])             # add with weight
                        model[cla][feature_idx][val] = float(cnt + laplacian) / (total + len(possible_vals[feature_idx])*laplacian)
                else:                            # if continuous feature
                    c = len(index)
                    local_reg = weight[index]
                    local_reg /= total
                    local_reg *= c
                    vals = local_reg * dataset[index, feature_idx]
                    mu = np.mean(vals)
                    sigma = math.sqrt(np.var(vals))
                    model[cla][feature_idx]['mu'] = mu
                    model[cla][feature_idx]['sigma'] = sigma

        return model, class_prob

    def adaboost_weight(self, dataset, labels, catetype, possible_vals, T):

        m, d = dataset.shape
        MODELS = []        # store classifiers
        CLASS_PROBS = []   # store classifiers
        ALPHA = []
        W = np.array([1.0 / m for _ in range(m)])
        t = 0
        while True:
            t += 1
            model, class_prob = self.get_model_weight(dataset, catetype, labels, possible_vals, W)
            err_label, err_rate = self.get_err_rate_weight(dataset, labels, W, model, class_prob, catetype)
            if err_rate >= 0.5 or err_rate < 1e-7 or t > T:
                break
            alpha = 0.5 * math.log((1 - err_rate) / err_rate)
            MODELS.append(model)  # Store Model
            CLASS_PROBS.append(class_prob)
            ALPHA.append(alpha)
            for i in range(m):
                if err_label[i] == 1:  # misclassified instance
                    W[i] *= math.exp(alpha)
                else:
                    W[i] *= math.exp(-alpha)
            sumW = sum(W)
            for i in range(m):
                W[i] /= sumW
        return MODELS, CLASS_PROBS, ALPHA

    def get_belong(self, PREDICT, clas):
        mini = float(1 << 30)
        belong = 0
        for cla in clas:
            if abs(cla - PREDICT) < mini:
                mini = abs(cla - PREDICT)
                belong = cla
        return belong

    def adaboost_work(self, w, T=10):   # w is True: weight-embedding, w is False: re-sampling
        accs = []
        for trainset, trainlabels, testset, testlabels in self.gen_CV(self.dataset, self.labels, 5):

            if w:
                MODELS, CLASS_PROBS, ALPHA = self.adaboost_weight(trainset, trainlabels, self.catetype, self.possible_vals, T)
            else:
                MODELS, CLASS_PROBS, ALPHA = self.adaboost(trainset, trainlabels, self.catetype, self.possible_vals, T)
            m, d = testset.shape
            clas = set(self.labels)
            err = 0
            for i in range(m):
                PREDICT = 0.0
                for model, class_prob, alpha in zip(MODELS, CLASS_PROBS, ALPHA):
                    predict = self.classify(testset[i], model, class_prob, self.catetype)
                    PREDICT += alpha * predict
                belong = self.get_belong(PREDICT, clas)
                if belong != testlabels[i]:
                    err += 1
            err_rate = float(err) / m
            accs.append(1.0 - err_rate)
        mean = np.mean(accs)
        var = np.var(accs)
        print("{} base classifiers:[{}] -> Mean: {:.4f}, Variance: {:.6f}".format(T, "WEIGHT" if w else "ADABOOST", mean, var))
        return mean, var

if __name__ == '__main__':

    nb = NaiveBayes('breast-cancer-assignment5.txt')
    base_mean, base_var = nb.work()
    MEAN = []
    VAR = []
    for T in range(1, 20):
        mean, var = nb.adaboost_work(True, T)   # weight-embedding
        MEAN.append(mean * 100)
        VAR.append(var)

    plt.xlabel("Num of base classifier")
    plt.ylabel("Mean Accuracy(%)")
    plt.ylim(0, 100)
    plt.plot([i for i in range(20)], [base_mean*100 for i in range(20)], linestyle='-', color='g', marker='x', linewidth=1.2, label="One NB Classifier")
    plt.plot(MEAN, linewidth=1.2, color='r', label="Adaboost")
    plt.legend(loc='upper right')
    plt.show()

    # for T in range(1, 10):
    #     nb.adaboost_work(False, T)  # re-sampling

    print("===========================================")

    nb2 = NaiveBayes('german-assignment5.txt')
    base_mean, base_var = nb2.work()
    MEAN = []
    VAR = []
    for T in range(1, 21):
        mean, var = nb2.adaboost_work(True, T)
        MEAN.append(mean * 100)
        VAR.append(var)
    plt.xlabel("Num of base classifier")
    plt.ylabel("Mean Accuracy(%)")
    plt.ylim(0, 100)
    plt.plot([i for i in range(20)], [base_mean * 100 for i in range(20)], linestyle='-', color='g', marker='x',
             linewidth=1.2, label="One NB Classifier")
    plt.plot(MEAN, linewidth=1.2, color='r', label="Adaboost")
    plt.legend(loc='upper right')
    plt.show()



















