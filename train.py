# coding=utf-8
import data_loader
import network

file_path = 'data/model.json'


def train():
    training_data, validation_data = data_loader.load_data_for_training()
    layers = [784, 100, 10]
    net = network.Network(layers)
    net.optimize(training_data, validation_data, 30, 5, 10, 0.1, reg_lambda=5.0)
    net.save(file_path)
    print('Model saved at %s' % file_path)


if __name__ == '__main__':
    train()
