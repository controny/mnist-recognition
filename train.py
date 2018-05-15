# coding=utf-8
import data_loader
import network


def train():
    training_data, validation_data, test_data = data_loader.load_all_data()
    layers = [784, 100, 10]
    net = network.Network(layers)
    net.optimize(training_data, validation_data, 30, 1, 10, 1.0)
    # 测试
    num_correct = net.accuracy(test_data)
    accuracy = 1.0 * num_correct / len(test_data)
    print('Final: %f (%d / %d)' % (accuracy, num_correct, len(test_data)))


if __name__ == '__main__':
    train()
