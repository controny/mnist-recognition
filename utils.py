# coding=utf-8
import numpy as np


def sigmoid(x):
    """计算Sigmoid(x)"""
    return 1.0/(1.0+np.exp(-x))


def sigmoid_derivative(x):
    """计算Sigmoid对于输入x的导数"""
    return sigmoid(x)*(1-sigmoid(x))


def cross_entropy(a, y):
    """求交叉熵并求和"""
    # np.nan_to_num保证结果不为NaN
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


def cross_entropy_derivative(a, y):
    """求交叉熵的导数"""
    return a - y
