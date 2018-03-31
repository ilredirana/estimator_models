
import os
from functools import partial
from urllib import request as ur

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def get_np_image_from_path(image_path, height=299, width=299, channels=3):
    return get_np_image_input(np.asarray(cv2.imread(image_path)), height=height, width=width, channels=channels)


def get_np_image_from_url(image_url, timeout=5, height=299, width=299, channels=3):
    image_string = ur.urlopen(image_url, timeout=timeout).read()
    np_arr = np.fromstring(image_string, dtype=np.uint8)
    return get_np_image_input(cv2.imdecode(np_arr, cv2.IMREAD_COLOR), height=height, width=width, channels=channels)


def get_np_image_from_bytes(image_bytes, height=299, width=299, channels=3):
    np_arr = np.fromstring(image_bytes, dtype=np.uint8)
    return get_np_image_input(cv2.imdecode(np_arr, cv2.IMREAD_COLOR), height=height, width=width, channels=channels)


def get_np_image_input(image_tensor=None, height=299, width=299, channels=3, normalize=True):
    """
    返回一个shape为[1, height, width, channels]的numpy张量
    :param image_tensor:
    :param height:
    :param width:
    :param channels:
    :return:
    """
    image = central_crop(image_tensor, 1)
    image = cv2.resize(image, (height, width))
    if normalize:
        image = normalize_image(image)
    image = image.astype(np.float32)
    return image.reshape(-1, height, width, channels)


def normalize_image(x):
    """
    归一化
    :param x:
    :return:
    """
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


def central_crop(img, rate):
    """
    居中裁剪
    :param img:
    :param rate:
    :return:
    """
    if rate <= 0.0 or rate > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    y, x, h = img.shape
    if y == x and rate == 1:
        return img
    residual_hw = int(min(x, y) * rate)
    start_x = x // 2 - (residual_hw // 2)
    start_y = y // 2 - (residual_hw // 2)
    return img[start_y:start_y + residual_hw, start_x:start_x + residual_hw]


# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def load_image_tf(filename, label, height, width, channels=3, normalize=True):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels)
    image_decoded.set_shape([None, None, None])
    image_central_cropped = tf.image.central_crop(image_decoded, 1)
    image_resized = tf.image.resize_images(image_central_cropped, tf.constant([height, width], tf.int32),
                                           method=ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_resized, height, width)
    image_resized = tf.reshape(image_resized, [height, width, channels])
    if normalize:
        image_resized = tf.divide(image_resized, 255.0)
        image_resized = tf.subtract(image_resized, 0.5)
        image_resized = tf.multiply(image_resized, 2.0)
    image_resized = tf.to_float(image_resized)
    return image_resized, image_decoded, label


def read_folder(folders, labels):
    if not isinstance(folders, (list, tuple, set)):
        raise ValueError("folders 应为list 或 tuple")
    if not isinstance(labels, (list, tuple, set)):
        raise ValueError("labels 应为list 或 tuple")
    all_files = []
    all_labels = []
    for i, f in enumerate(folders):
        files = os.listdir(f)
        for file in files:
            all_files.append(os.path.join(f, file))
            all_labels.append(labels[i])
    dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
    return dataset, len(all_files)


def dataset_input_fn(folders, labels, epoch, batch_size,
                     height, width, channels,
                     scope_name="dataset_input",
                     feature_name=None, origin_images=False):
    def fn():
        with tf.name_scope(scope_name):
            dataset, length = read_folder(folders, labels)
            dataset = dataset.map(partial(load_image_tf, height=height, width=width, channels=channels))
            dataset = dataset.shuffle(buffer_size=length).repeat(epoch).batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            one_element = iterator.get_next()
            if feature_name and origin_images:
                return {str(feature_name): one_element[0], "origin_images": one_element[1]}, one_element[2]
            if feature_name:
                return {str(feature_name): one_element[0]}, one_element[2]
            return one_element[0], one_element[2]
    return fn
