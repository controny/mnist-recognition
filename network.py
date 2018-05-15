# coding=utf-8
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        初始化神经网络
        :param sizes: 表示网络的结构，如[3, 2, 1]表示三层网络，每一层分别有3,2,1个神经元
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 随机初始化参数
        # 第1层为输入，故从sizes[1]开始初始化
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]]
        self.weights = [np.random.randn(j, i)
                        for i, j in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, x):
        """
        前向传播计算最终输出
        :param x: 模型的输入
        :return: 模型的输出
        """
        for w, b in zip(self.weights, self.biases):
            # 以Sigmoid作为激活函数
            x = sigmoid(np.dot(w, x) + b)

        return x


def sigmoid(z):
    """给定输入z，计算Sigmoid(z)"""
    return 1.0/(1.0+np.exp(-z))
