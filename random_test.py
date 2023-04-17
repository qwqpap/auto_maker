import numpy as np
import cv2
from numba import jit
import random

front = cv2.imread("target.jpg")
back = cv2.imread('back.jpg')


@jit
def mix_pic(img, img_back, loca_y, loca_x):
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
    #one_normal = np.random.normal(loc=1.0, scale=0.2, size=(1, 4))
    one_normal = np.random.random_sample(size=(1, 4))
    a = one_normal[0, 0]
    d = one_normal[0, 1]
    b = one_normal[0, 2]
    c = one_normal[0, 3]

    # target_img = cv2.imread('target.jpg')
    # back_img = cv2.imread('back.jpg')
    target_img = cv2.resize(target_img, (500, 300))
    cols, rows = target_img.shape[:2]
    half_cols = cols/2
    half_rows = rows/2
    point_left_up = 0

    basis_vector = min(cols, rows)



    dst = 0
    position_x = 0
    position_y = 0

    fina = mix_pic(dst, back_img, position_y, position_x)
    # back_img[0:af_cols, 0:af_rows] = dst
    cv2.imshow('fin', fina)
    yolo_format = []
    # print(position_x)
    # print(position_y)
    # print(rand_x)
    # print(rand_y)

    # print(yolo_format)
    cv2.waitKey(0)
    return fina, yolo_format


img_mix(front, back)