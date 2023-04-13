import pyautogui as gui
import cv2
import numpy
import os
import time

import cv2
import numpy as np
import random
import numba
from numba import jit


@jit
def mix_pic(img, img_back, loca_x, loca_y):
    # loca_x = 10
    # loca_y = 10

    for x in range(img.shape[0]):  # 图片的高
        for y in range(img.shape[1]):  # 图片的宽
            px = img[x, y]
            if px[0] == 0 and px[1] == 0 and px[2] == 0:
                pass
            else:
                img_back[loca_x + x, loca_y + y] = px
                # print('1')
    return img_back


def img_mix(target_img, back_img):

    one_normal = np.random.normal(loc=1.0, scale=0.3, size=(1, 2))
    zero_normal = np.random.normal(loc=0.0, scale=0.2, size=(1, 2))
    a = one_normal[0, 0]
    d = one_normal[0, 1]
    b = zero_normal[0, 0]
    c = zero_normal[0, 1]

    # target_img = cv2.imread('target.jpg')
    # back_img = cv2.imread('back.jpg')
    target_img = cv2.resize(target_img, (460, 350))
    cols, rows = target_img.shape[:2]
    basis_vector = max(cols, rows)

    # print(transfer_x)

    # 传递小于0
    anti_c = (abs(c) - c) / 2
    anti_b = (abs(b) - b) / 2
    x_delta = rows * anti_c
    y_delta = cols * anti_b

    rows = int((a * rows + c * rows) + (2 * x_delta))
    cols = int((b * cols + d * cols) + (2 * y_delta))
    a = int(a * basis_vector)
    b = int(b * basis_vector)
    c = int(c * basis_vector)
    d = int(d * basis_vector)

    a = a + x_delta
    b = b + y_delta
    c = c + x_delta
    d = d + y_delta
    transfer_x = [a, b]
    transfer_y = [c, d]

    pts1 = np.float32([[0, 0], [basis_vector, 0], [0, basis_vector]])
    pts2 = np.float32([[x_delta, y_delta], transfer_x, transfer_y])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(target_img, M, (rows, cols))
    front_x, front_y = dst.shape[:2]

    back_img = cv2.resize(back_img, (1920, 1080))
    rand_x = random.randint(200, 600)
    rand_y = int(rand_x * (front_y / front_x))
    dst = cv2.resize(dst, (rand_x, rand_y))
    position_x = random.randint(0, 1980 - rand_x - 1)
    position_y = random.randint(0, 1080 - rand_y - 1)

    fina = mix_pic(dst, back_img, rand_x, rand_y)
    # back_img[0:af_cols, 0:af_rows] = dst
    cv2.imshow('fin', fina)
    yolo_format = []
    yolo_format.append((position_x + rand_x / 2) / 1920)
    yolo_format.append((position_y + rand_y / 2) / 1080)
    yolo_format.append(rand_x / 1920)
    yolo_format.append(rand_y / 1080)
    print(yolo_format)

    cv2.waitKey(0)


class AutoMakerDataYolo:
    '''
    this is a class to automatic making the data of yolov5
    这是一个自动化生成yolov5数据集的库
    '''

    def __init__(self):
        self.target_number = None
        self.target_name = None
        self.target_image_list = None
        self.made_number = None
        self.target_location = None
        self.huge_location = None
        self.num = 1
        self.yse_or_no = gui.confirm(text='make sure the name of target image is "class.png"', title='setup runing',
                                     buttons=['yes', 'no'])
        if self.yse_or_no == 'no':
            quit()

    def got_data(self):
        self.huge_location = gui.prompt(text='the path of background image', title='qwq', default='/data/image/')
        self.target_location = gui.prompt(text='the path of target image', title='qwq', default='/data/target/')
        self.made_number = gui.prompt(text='the number you want for each target image', title='qwq', default='114514')

    def read_target(self):
        # 读取目标的数量种类，存在列表里
        image_list = os.listdir(self.target_location)
        self.target_name = []
        self.target_image_list = []
        for image_name in image_list:
            if image_name.endswith('.jpg') or image_name.endswith('png'):
                image = cv2.imread(self.huge_location + image_name)
                self.target_image_list.append(image)
                self.target_name.append(image_name)
            else:
                pass
        self.target_number = len(self.target_name)


    def read_image(self):
        image_list = os.listdir(self.target_location)
        for image_name in image_list:
            if image_name.endswith('.jpg') or image_name.endswith('png'):
                image = cv2.imread(self.target_location + image_name)




AUTO = AutoMakerDataYolo()
AUTO.got_data()
