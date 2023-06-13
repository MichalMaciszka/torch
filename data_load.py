import os
import random
import numpy as np
from constants import *
import cv2
from PIL import Image


def load_data(p):
    image_names = os.listdir(p)
    paths = []
    for i in image_names:
        paths.append(f"{p}/{i}")
    data = []
    for path in paths:
        data.append(cv2.imread(path))
    print(f"Data loaded for {p}")
    return data


def scale(img):
    color = [255, 255, 255]
    new_img = cv2.copyMakeBorder(img, 0, 136-102, 0, 0, cv2.BORDER_CONSTANT, value=color)
    return new_img


def scale_images(data):
    to_return = []
    for i in data:
        to_return.append(scale(i))
    return to_return


def norm_image(image):
    img = np.asarray(image).astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    img = (image - min_val) / (max_val - min_val)
    return img


def norm_images(data):
    to_return = []
    for i in data:
        to_return.append(norm_image(i))
    return to_return


def convert_image_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_all_to_grayscale(data):
    to_return = []
    for i in data:
        to_return.append(convert_image_to_grayscale(i))
    return to_return


def get_dataset(boots, shoes, sandals):
    dataset = []
    for x in boots:
        dataset.append((x, CLASSES["boot"]))
    for x in shoes:
        dataset.append((x, CLASSES["shoe"]))
    for x in sandals:
        dataset.append((x, CLASSES["sandal"]))

    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    val_set = dataset[0::10]
    test_set = dataset[1::10]

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    x_train_set = [e[0] for e in train_set]
    y_train_set = [e[1] for e in train_set]
    x_val_set = [e[0] for e in val_set]
    y_val_set = [e[1] for e in val_set]
    x_test_set = [e[0] for e in test_set]
    y_test_set = [e[1] for e in test_set]

    x_train_set = np.array(x_train_set, dtype=np.uint8)
    y_train_set = np.array(y_train_set, dtype=np.uint8)
    x_val_set = np.array(x_val_set, dtype=np.uint8)
    y_val_set = np.array(y_val_set, dtype=np.uint8)
    x_test_set = np.array(x_test_set, dtype=np.uint8)
    y_test_set = np.array(y_test_set, dtype=np.uint8)

    return x_train_set, x_test_set, x_val_set, y_train_set, y_test_set, y_val_set
