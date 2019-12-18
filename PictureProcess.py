import os
import numpy as np
import cv2
import random


def picture_process(train):
    datas = []
    labels = []
    fpaths = []
    path = 'D:\\python\\Tensorflow-CNN-Tutorial\\data\\'
    if not train:
        path = 'D:\\python\\Tensorflow-CNN-Tutorial\\data\\yzc_test\\'
        for img_name in os.listdir(path):
            img_path = path + img_name
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            img_raw = np.array(img) / 255.0
            datas.append(img_raw)
            # labels.append(random.randint(0,2))
            labels.append(2)
            fpaths.append(img_name)
        print(fpaths)
        return fpaths, datas, labels

    for dir_name in os.listdir(path):
        print(dir_name)
        if dir_name == 'yzc_test':
            continue
        curPath = path + dir_name + "\\"
        if os.path.isdir(curPath):
            for img_name in os.listdir(curPath):
                img_path = curPath + img_name
                img = cv2.imread(img_path)
                img = cv2.resize(img, (32, 32))
                img_raw = np.array(img) / 255.0
                datas.append(img_raw)
                label = 0 if dir_name == 'car' else 1 if dir_name == 'gc' else 2
                labels.append(label)
                fpaths.append(img_name)
    # data = np.array(datas)
    # print(data)
    print(fpaths)
    print(labels)
    return fpaths, datas, labels


if __name__ == '__main__':
    picture_process(True)
