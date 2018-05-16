# coding=utf-8
import data_loader
import network
import sys
import getopt
import os
from train import log_dir


def test(net, test_data):
    print('Train error: %f' % net.train_error)
    num_correct = net.accuracy(test_data)
    accuracy = 1.0 * num_correct / len(test_data)
    print('Accuracy: %f (%d / %d)' % (accuracy, num_correct, len(test_data)))
    print('Test error: %f' % net.total_loss(test_data))


def main():
    model_name = 'model'
    # 设置命令行参数更改model_name
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['model_name='])
    except getopt.GetoptError as err:
        print('ERROR: %s!' % err)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--model_name':
            model_name = arg

    # 载入数据和模型
    test_data = data_loader.load_data_for_testing()
    model_path = os.path.join(log_dir, model_name+'.json')
    net = network.Network.load(model_path)

    test(net, test_data)


if __name__ == '__main__':
    main()
