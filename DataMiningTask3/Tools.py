#! usr/bin/python
# coding=utf-8
"""
File Name: Tools.py
Description: Calc Gini or Purity of a Clustering Result
Date: 2016-10-30
Author: QIU HU
"""
import numpy as np
import copy


def pl(lis):
    """
    print a list, one by a line
    :param lis: list need to be print
    :return: None
    """
    for li in lis:
        print(li)


def pl2d(lis):
    """
    print a 2-D list, one list by a line
    :param lis:
    :return: None
    """
    for i in range(len(lis)):
        print(" ".join(map(str, lis[i])))


def read_dataset(data):    # Read dataset for processing

    datamat = np.loadtxt(data, delimiter=',')
    dataSet = datamat[:, :-1]
    label = datamat[:, -1]
    groundtruth = sorted(list(set(label)))  # [-1, 1]
    return dataSet, label, groundtruth


def calc_GINI(M):

    Mcp = copy.deepcopy(M)
    sigma_M = np.sum(M, axis=0)
    tmp = (Mcp / sigma_M) ** 2
    gj = np.sum(tmp, axis=0)
    Gj = map(lambda x: 1-x, gj)
    Gaverage = sum(Gj * sigma_M) / sum(sigma_M)
    return Gaverage


def calc_Purity(M):    # M is an array

    sigma_M = np.sum(M, axis=0)
    P = np.max(M, axis=0)
    return float(sum(P)) / sum(sigma_M)


def Distance(dataSet, i, c):

    return sum(abs(dataSet[i]-dataSet[c]))

# if __name__ == '__main__':
#
#     a = np.array([[1, 2, -1], [3, 4, -2]])
#     print(Distance(a, 0, 1))
