# coding=utf-8
import data_loader
import network


def train():
    training_data, validation_data, test_data = data_loader.load_all_data()
    # layers = [784, 100, 100, 10]
    layers = [784, 20, 10]
    net = network.Network(layers)
    net.optimize(training_data, validation_data, 30, 1, 100, 0.005)


if __name__ == '__main__':
    train()
