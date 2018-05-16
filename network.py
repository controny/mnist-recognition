# coding=utf-8
import numpy as np
import utils
import json


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
        self.weights = [np.random.randn(j, i)/np.sqrt(i) for i, j in zip(sizes[:-1], sizes[1:])]
        self.train_error = 0.0
        self.reg_lambda = 0.0

    def feed_forward(self, x):
        """
        前向传播，用于back propagation计算梯度
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

    def optimize(self, training_data, validation_data, epochs, validating_gap, batch_size, learning_rate,
                 reg_lambda=0.0):
        """
        使用Stochastic Gradient Descent优化模型
        :param training_data: 训练数据集
        :param validation_data: 验证数据集
        :param epochs: 迭代次数，每遍历一次训练数据算作一次迭代
        :param validating_gap: 验证的间隔步数
        :param batch_size: 每个训练batch的大小
        :param learning_rate: 训练的学习率
        :param reg_lambda: 正则化项的lambda参数，默认为0表示不加入正则化项
        """
        self.reg_lambda = reg_lambda
        for i in range(epochs):
            # 随机打乱训练数据
            np.random.shuffle(training_data)
            # 分批次训练
            for k in range(0, len(training_data), batch_size):
                batch = training_data[k:k+batch_size]
                self.update_parameters(batch, learning_rate)
            # 验证准确率
            if i % validating_gap == 0:
                num_correct = self.accuracy(validation_data)
                accuracy = 1.0 * num_correct / len(validation_data)
                print('Epoch %d: %f (%d / %d)' % (i, accuracy, num_correct, len(validation_data)))
            else:
                print('Epoch %d: finished' % i)
        self.train_error = self.total_loss(validation_data)
        print('Train error: %f' % self.train_error)

    def update_parameters(self, batch, learning_rate):
        """
        根据一个训练的batch，利用back propagation计算参数梯度并更新参数
        :param batch: 一个训练batch
        :param learning_rate: 训练的学习率
        """
        # 累积的梯度
        sum_delta_weights = [np.zeros(e.shape) for e in self.weights]
        sum_delta_biases = [np.zeros(e.shape) for e in self.biases]
        # 对于每个样本输入都计算梯度，并记录下来
        for x, y in batch:
            delta_weights, delta_biases = self.back_propagation(x, y)
            sum_delta_weights = [s+d for s, d in zip(sum_delta_weights, delta_weights)]
            sum_delta_biases = [s+d for s, d in zip(sum_delta_biases, delta_biases)]
        # 根据上述梯度的平均值更新参数
        batch_size = len(batch)
        to_subtract_weights = [learning_rate*s/batch_size for s in sum_delta_weights]
        to_subtract_biases = [learning_rate*s/batch_size for s in sum_delta_biases]
        # weights的更新考虑L2正则化
        self.weights = [(1-learning_rate*self.reg_lambda/batch_size)*w-t
                        for w, t in zip(self.weights, to_subtract_weights)]
        self.biases = [b-t for b, t in zip(self.biases, to_subtract_biases)]

    def back_propagation(self, x, y):
        """
        根据单个训练样本计算梯度
        :param x: 样本的输入
        :param y: 样本的标签
        :return: 一个tuple，包括weights和biases的梯度
        """
        zs, activations = self.feed_forward(x)
        delta_weights = [np.zeros(e.shape) for e in self.weights]
        delta_biases = [np.zeros(e.shape) for e in self.biases]
        # 计算输出层的误差
        delta = utils.cross_entropy_derivative(activations[-1], y) * utils.sigmoid_derivative(zs[-1])
        # 后向传播误差
        delta_weights[-1] = np.dot(delta, activations[-2].transpose())
        delta_biases[-1] = delta
        for last in range(2, self.num_layers):
            z = zs[-last]
            delta = np.dot(self.weights[-last+1].transpose(), delta) * utils.sigmoid_derivative(z)
            delta_weights[-last] = np.dot(delta, activations[-last-1].transpose())
            delta_biases[-last] = delta

        return delta_weights, delta_biases

    def predict(self, x):
        """
        给定输入，给出模型的预测结果
        :param x: 模型的输入，
        :return: 模型的输出
        """
        for w, b in zip(self.weights, self.biases):
            x = utils.sigmoid(np.dot(w, x) + b)

        return x

    def accuracy(self, data):
        """
        给定测试数据，计算模型预测正确的个数
        :param data: 用于测试的验证数据集或测试数据集
        :return: 预测正确的个数
        """
        results = [(np.argmax(self.predict(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_loss(self, data):
        """
        给定测试数据，计算模型的loss
        :param data: 用于测试的验证数据集或测试数据集
        :return: 模型的loss
        """
        loss = 0.0
        for x, y in data:
            a = self.predict(x)
            loss += utils.cross_entropy(a, y) / len(data)
        # 加上L2正则化项
        loss += 0.5*(self.reg_lambda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)

        return loss

    def save(self, model_path):
        """
        保存模型为json格式
        :param model_path: 保存的文件路径
        """
        data = {
            'sizes': self.sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'train_error': self.train_error,
            'reg_lambda': self.reg_lambda
        }
        with open(model_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load(model_path):
        """
        用于从文件中载入模型的静态方法
        :param model_path: 模型文件的路径
        :return: 载入的模型
        """
        with open(model_path, 'r') as f:
            data = json.load(f)
        net = Network(data['sizes'])
        net.weights = [np.array(w) for w in data['weights']]
        net.biases = [np.array(b) for b in data['biases']]
        net.train_error = data['train_error']
        net.reg_lambda = data['reg_lambda']

        return net
