# coding=utf-8
from PIL import Image
import sys
import os
import numpy as np
import network
from train import log_dir


def inference():
    images_dir = 'images/'
    # 设置命令行参数更改图片目录
    if len(sys.argv) == 2:
        images_dir = sys.argv[1]

    # 载入模型
    model_name = 'model'
    model_path = os.path.join(log_dir, model_name+'.json')
    net = network.Network.load(model_path)

    file_list = os.listdir(images_dir)
    for file_path in file_list:
        # 打开图片并转为灰度图
        image = Image.open(os.path.join(images_dir, file_path)).convert('L')
        image = image.resize((28, 28))
        # 转化为numpy array并进行二值化处理
        image = np.reshape(image, [784, 1])
        image = np.clip(image, 0.0, 1.0)
        prediction = np.argmax(net.predict(image))
        print('%s: %s' % (file_path, prediction))


if __name__ == '__main__':
    inference()
