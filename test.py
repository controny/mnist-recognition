# coding=utf-8
import data_loader
import network
import train


def test():
    # 载入数据和模型
    test_data = data_loader.load_data_for_testing()
    net = network.Network.load(train.file_path)
    print('Model loaded')
    print('Train error: %f' % net.train_error)

    num_correct = net.accuracy(test_data)
    accuracy = 1.0 * num_correct / len(test_data)
    print('Accuracy: %f (%d / %d)' % (accuracy, num_correct, len(test_data)))
    print('Test error: %f' % net.total_loss(test_data))


if __name__ == '__main__':
    test()
