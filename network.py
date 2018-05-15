# coding=utf-8
import numpy as np
import utils


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
        self.biases = np.asarray([np.random.randn(j, 1) for j in sizes[1:]])
        self.weights = np.asarray([np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])])

    def feed_forward(self, x):
        """
        前向传播，用于back propagation计算梯度，也用于得出模型的预测结果
        :param x: 模型的输入
        :return: 一个tuple，包括模型中所有的z值和activation
        """
        activation = x
        # 输入层算入activation，但不算入z值
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            # 以Sigmoid作为激活函数
            activation = utils.sigmoid(z)
            activations.append(activation)

        return zs, activations

    def optimize(self, training_data, validation_data, epochs, validating_gap, batch_size, learning_rate):
        """
        使用Stochastic Gradient Descent优化模型
        :param training_data: 训练数据集
        :param validation_data: 验证数据集
        :param epochs: 迭代次数，每遍历一次训练数据算作一次迭代
        :param validating_gap: 验证的间隔步数
        :param batch_size: 每个训练batch的大小
        :param learning_rate: 训练的学习率
        """
        for i in range(epochs):
            # 随机打乱训练数据
            np.random.shuffle(training_data)
            # 分批次训练
            for k in range(0, len(training_data), batch_size):
                batch = training_data[k:k+batch_size]
                self.update_parameters(batch, learning_rate)
            if i % validating_gap == 0:
                print('Epoch %d: %f (%d / %d)')

    def update_parameters(self, batch, learning_rate):
        """
        根据一个训练的batch，利用back propagation计算参数梯度并更新参数
        :param batch: 一个训练batch
        :param learning_rate: 训练的学习率
        """
        # 累积的梯度
        sum_delta_weights = np.zeros(self.weights.shape)
        sum_delta_biases = np.zeros(self.biases.shape)
        # 对于每个样本输入都计算梯度，并记录下来
        for x, y in batch:
            delta_weights, delta_biases = self.back_propagation(x, y)
            sum_delta_weights += delta_weights
            sum_delta_biases += delta_biases
        # 根据上述梯度的平均值更新参数
        batch_size = len(batch)
        average_delta_weights = sum_delta_weights / batch_size
        average_delta_biases = sum_delta_biases / batch_size
        self.weights -= average_delta_weights
        self.biases -= average_delta_biases

    def back_propagation(self, x, y):
        """
        根据单个训练样本计算梯度
        :param x: 样本的输入
        :param y: 样本的标签
        :return: 一个tuple，包括weights和biases的梯度
        """
        zs, activations = self.feed_forward(x)
        delta_weights = np.zeros(self.weights.shape)
        delta_biases = np.zeros(self.biases.shape)
        # 计算输出层的误差
        delta = utils.cross_entropy_derivative(activations[-1], y) * utils.sigmoid_derivative(zs[-1])
        # 后向传播误差
        delta_weights[-1] = delta
        delta_biases[-1] = np.dot(delta, activations[-2].transpose())
        for last in range(2, self.num_layers):
            z = zs[-last]
            delta = np.dot(self.weights[-last+1].transpose(), delta) * utils.sigmoid_derivative(z)
            delta_biases[-last] = delta
            delta_weights[-last] = np.dot(delta, activations[-last-1].transpose())

        return delta_weights, delta_biases
