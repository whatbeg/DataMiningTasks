#! usr/bin/python
# coding=utf-8
"""
File Name: Data Operation
Description:
Date: 2016-10-13
Author: QIU HU
"""



import numpy as np
import copy
import Tools


a = np.array([[1.0, 2.0, 3], [3, 4, 5.0], [5, 3, 2]])

print(a * -1)

sigma = np.sum(a, axis=0)
print(sigma)

maxi = np.max(a, axis=0)
print(maxi)



dataSet = np.loadtxt('german.txt', delimiter=',')
print(dataSet)

label = dataSet[:, -1]
# print(label)

dataSet = dataSet[:, :-1]
print(dataSet)

s = set(label)
print(s)
