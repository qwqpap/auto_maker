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

    # print(px)


def img_mix(target_img, back_img):
    one_normal = np.random.normal(loc=1.0, scale=0.3, size=(1, 2))
    zero_normal = np.random.normal(loc=0.0, scale=0.2, size=(1, 2))
    a = one_normal[0, 0]
    d = one_normal[0, 1]
    b = zero_normal[0, 0]
    c = zero_normal[0, 1]

    #target_img = cv2.imread('target.jpg')
    #back_img = cv2.imread('back.jpg')
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
    rand_y = int(rand_x * (front_y/front_x))
    dst = cv2.resize(dst,(rand_x,rand_y))
    position_x = random.randint(0,1980 - rand_x - 1)
    position_y = random.randint(0, 1080 - rand_y - 1)

    fina = mix_pic(dst, back_img, rand_x, rand_y)
    # back_img[0:af_cols, 0:af_rows] = dst
    cv2.imshow('fin', fina)
    yolo_format = []
    yolo_format.append((position_x+rand_x/2)/1920)
    yolo_format.append((position_y+rand_y/2)/1080)
    yolo_format.append(rand_x/1920)
    yolo_format.append(rand_y/1080)
    print(yolo_format)

    cv2.waitKey(0)
