# coding=utf-8
import data_loader
import network
import sys
import getopt
import os

log_dir = 'log/'


def train(net, epochs, validating_gap, batch_size, learning_rate, reg_lambda):
    training_data, validation_data = data_loader.load_data_for_training()
    print('Start to train')
    net.optimize(training_data, validation_data, epochs, validating_gap, batch_size, learning_rate, reg_lambda)


def save_model(net, model_name):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    model_path = os.path.join(log_dir, model_name+'.json')
    net.save(model_path)
    print('Model saved at %s' % model_path)


def main():
    epochs = 30
    validating_gap = 5
    batch_size = 10
    learning_rate = 1.0
    reg_lambda = 0.0
    model_name = 'model'
    # 处理命令行参数
    try:
        opts, args = getopt.getopt(sys.argv[1:], '',
                                   ['epochs=',
                                    'validating_gap=',
                                    'batch_size=',
                                    'learning_rate=',
                                    'reg_lambda=',
                                    'model_name='
                                    ])
    except getopt.GetoptError as err:
        print('ERROR: %s!' % err)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--epochs':
            epochs = int(arg)
        elif opt == '--validating_gap':
            validating_gap = int(arg)
        elif opt == '--batch_size':
            batch_size = int(arg)
        elif opt == '--learning_rate':
            learning_rate = float(arg)
        elif opt == '--reg_lambda':
            reg_lambda = float(arg)
        elif opt == '--model_name':
            model_name = arg

    layers = [784, 100, 10]
    net = network.Network(layers)
    train(net, epochs, validating_gap, batch_size, learning_rate, reg_lambda)
    save_model(net, model_name)


if __name__ == '__main__':
    main()
