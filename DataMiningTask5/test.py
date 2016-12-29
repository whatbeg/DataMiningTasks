#! usr/bin/python
# coding=utf-8
"""
File Name: NaiveBayes.py
Description: Naive Bayes Classifier
Date: 2016-11-19
Author: QIU HU
"""

import numpy as np
import math


a = np.array([[1, 2, 3],
              [4, 5, 6],
              [1, 2, 6]])
labels = a[:, -1]

weight = np.array([1, 2, 3])

index = np.where(a[0, :] == 2)

t = np.sum(weight[index] * a[0, index])

logi = weight[index]

sss = [0.65, 0.72, 0.71, 0.72, 0.64, 0.73, 0.77, 0.6699999999999999, 0.6699999999999999, 0.72]

print(np.mean(sss))
