#! usr/bin/python
# coding=utf-8
"""
File Name: Tools.py
Description: Some Functions
Date: 2016-11-05
Author: QIU HU
"""
import math
import numpy as np


def sigmoid(val):
    return 1.0 / (1.0 + math.exp(-val))


def sign(val):
    if val > 1e-7:
        return 1
    elif val < -1e-7:
        return -1
    return 0


def classify(w, inX):
    probabilty = sigmoid(np.dot(w, inX))
    # print(probabilty)
    if probabilty > 0.5:
        return 1.0
    else:
        return -1.0


def classify_ridge(w, inX):
    prob = np.dot(w, inX)
    if prob > 0.0:
        return 1.0
    else:
        return -1.0

