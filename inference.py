# coding=utf-8
from PIL import Image
import sys
import os
import getopt
import numpy as np
import network
from train import log_dir


def inference(net, images_dir):
    file_list = os.listdir(images_dir)
    print('Picture\t\t\tPrediction')
    for file_path in file_list:
        # 打开图片并转为灰度图
        image = Image.open(os.path.join(images_dir, file_path)).convert('L')
        image = image.resize((28, 28))
        # 转化为numpy array并进行二值化处理
        image = np.reshape(image, [784, 1])
        image = np.clip(image, 0.0, 1.0)
        prediction = np.argmax(net.predict(image))
        print('%s:\t\t\t%s' % (file_path, prediction))


def main():
    model_name = 'model'
    images_dir = 'images/'
    # 设置命令行参数更改model_name及images_dir
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['model_name=', 'images_dir='])
    except getopt.GetoptError as err:
        print('ERROR: %s!' % err)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--model_name':
            model_name = arg
        elif opt == '--images_dir':
            images_dir = arg

    # 载入模型
    model_path = os.path.join(log_dir, model_name+'.json')
    net = network.Network.load(model_path)

    inference(net, images_dir)


if __name__ == '__main__':
    main()
