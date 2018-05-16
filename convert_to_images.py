# coding=utf-8
import data_loader
from PIL import Image
import numpy as np
import os


def convert():
    """用于把MNIST的数据转为图片文件"""
    images_dir = 'images/'
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    num_images = 10
    images = data_loader.load_images(data_loader.test_images_path, num_images, binaryzation=False)
    extension = '.png'
    for index, image in enumerate(images):
        image = np.reshape(image, [28, 28])
        # 注意把image的数据类型转化为uint8，并以灰度图模式('L')生成图片
        image = Image.fromarray(np.uint8(image), 'L')
        image.save(os.path.join(images_dir, str(index)+extension))


if __name__ == '__main__':
    convert()
