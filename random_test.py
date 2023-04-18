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
    one_normal = np.random.normal(loc=0.5, scale=0.2, size=(1, 4))
    # one_normal = np.random.random_sample(size=(1, 4))
    a = one_normal[0, 0]
    d = one_normal[0, 1]
    b = one_normal[0, 2]
    c = one_normal[0, 3]
    a = (abs(a) + a) / 2
    b = (abs(b) + b) / 2
    c = (abs(c) + c) / 2
    d = (abs(d) + d) / 2

    target_img = cv2.resize(target_img, (500, 300))
    cols, rows = target_img.shape[:2]
    half_cols = cols/2
    half_rows = rows/2
    row_r_cols = cols/rows
    left_up_point = [int(half_cols - half_cols*a), int(half_rows - a*half_rows)]
    right_up_point = [int(half_cols + b*half_cols),int(half_rows - b*half_rows)]
    left_down_point = [int(half_cols - half_cols*c), int(half_rows + c*half_rows)]
    right_down_point = [int(half_cols + d*half_cols), int(half_rows + d*half_rows)]

    x_sharp = int(half_cols*(1-max(a,c)))
    y_sharp = int(half_rows*(1-max(a,b)))

    left_up_point[0] = left_up_point[0] - x_sharp
    left_up_point[1] = left_up_point[1] -y_sharp
    right_down_point[0] = right_down_point[0] - x_sharp
    right_down_point[1] = right_down_point[1] - y_sharp
    left_down_point[0] = left_down_point[0] - x_sharp
    left_down_point[1] = left_down_point[1] - y_sharp
    right_up_point[0] = right_up_point[0] - x_sharp
    right_up_point[1] = right_up_point[1] - y_sharp
    targ_col = int(half_cols*max(b,d)+half_cols*max(a,c))
    targ_row = int(half_rows*max(a,b)+half_rows*max(c,d))
    #print(left_up_point,left_down_point,right_up_point,right_down_point)
    point1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    # print(cols)
    # print(rows)
    point2 = np.float32([left_up_point, right_up_point, left_down_point, right_down_point])
    M = cv2.getPerspectiveTransform(point1, point2)
    dst = cv2.warpPerspective(target_img, M, (targ_col, targ_row))

    position_x = random.randint(0,(1920-cols))
    position_y = random.randint(0,(1080-rows))
    back_img = cv2.resize(back_img,(1920,1080))
    fina = mix_pic(dst, back_img, position_y, position_x)
    # back_img[0:af_cols, 0:af_rows] = dst
    cv2.imshow('fin', fina)
    yolo_format = []
    yolo_format.append((position_x + targ_col / 2) / 1920)
    yolo_format.append((position_y + targ_row / 2) / 1080)
    yolo_format.append((targ_col / 1920) * 0.8)
    yolo_format.append((targ_row / 1080) * 0.8)
    # print(position_x)
    # print(position_y)
    # print(rand_x)
    # print(rand_y)

    #print(yolo_format)
    cv2.waitKey(0)
    return fina, yolo_format



fina, ff = img_mix(front, back)
cv2.imshow('qwq', fina)
cv2.waitKey('0')