# coding=utf-8
import data_loader
import matplotlib.pyplot as plt
import numpy as np
import os


def convert():
    images_dir = 'images/'
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    num_images = 10
    images = data_loader.load_images(data_loader.test_images_path, num_images, binaryzation=True)
    extension = '.png'
    for index, image in enumerate(images):
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.savefig(os.path.join(images_dir, str(index)+extension))


if __name__ == '__main__':
    convert()
