# coding=utf-8
import struct
import numpy as np
import matplotlib.pyplot as plt

# 从训练数据集中划分出验证集
num_training_set = 50000
num_validation_set = 10000
num_test_set = 10000

training_images_path = 'data/train-images-idx3-ubyte'
training_labels_path = 'data/train-labels-idx1-ubyte'
validation_images_path = training_images_path
validation_labels_path = training_labels_path
test_images_path = 'data/t10k-images-idx3-ubyte'
test_labels_path = 'data/t10k-labels-idx1-ubyte'

images_data_header = '>IIII'
labels_data_header = '>II'
images_data_pattern = '>784B'
labels_data_pattern = '>1B'


def load_data_for_training():
    """
    载入训练所需的数据，包括训练集、验证集
    :return: 一个tuple，包括训练集、验证集
    """
    training_data = list(zip(
        load_images(training_images_path, num_training_set),
        load_labels(training_labels_path, num_training_set)
    ))
    validation_data = list(zip(
        load_images(validation_images_path, num_validation_set),
        load_labels(validation_labels_path, num_validation_set)
    ))

    return training_data, validation_data


def load_data_for_testing():
    """
    载入测试集
    :return: 测试集数据
    """
    test_data = list(zip(
        load_images(test_images_path, num_test_set),
        load_labels(test_labels_path, num_test_set)
    ))

    return test_data


def load_images(file_path, num_images):
    with open(file_path, 'rb') as f:
        buffer = f.read()
        # 跳过头部
        offset = struct.calcsize(images_data_header)
        images = []
        for i in range(num_images):
            image = struct.unpack_from(images_data_pattern, buffer, offset)
            # 注意进行二值化处理
            image = np.clip(np.reshape(image, [784, 1]), 0.0, 1.0)
            images.append(image)
            offset += struct.calcsize(images_data_pattern)

        return images


def load_labels(file_path, num_labels):
    with open(file_path, 'rb') as f:
        buffer = f.read()
        # 跳过头部
        offset = struct.calcsize(labels_data_header)
        labels = []
        for i in range(num_labels):
            digit = struct.unpack_from(labels_data_pattern, buffer, offset)[0]
            # 转化为one-hot vector
            label = np.zeros([10, 1])
            label[digit] = 1.0
            labels.append(label)
            offset += struct.calcsize(labels_data_pattern)

        return labels


def test():
    images = load_images(training_images_path, num_training_set)
    labels = load_labels(training_labels_path, num_training_set)
    for i in range(1):
        image = np.reshape(images[i], (28, 28))
        plt.imshow(image, cmap='gray')
        print('label:', labels[i])
        plt.show()


if __name__ == '__main__':
    test()
