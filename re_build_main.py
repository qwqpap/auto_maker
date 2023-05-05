import pyautogui as gui
import os
import time
import glob
import cv2
import numpy as np
import random
from numba import jit
import shutil
import re
import argparse

# 定义一点大家喜闻乐见的全局变量
huge_location = None
target_location = None
made_number = None
target_image_list = None
image_names_target = None


# pic mix part
@jit
def mix_pic(img, img_back, loca_x, loca_y):
    # 参数分别是前面图，后面背景图片，前面图片的左上角左边
    # 用于急速混合（真的很急速）
    for x in range(img.shape[0]):  # 图片的高
        for y in range(img.shape[1]):  # 图片的宽
            px = img[x, y]
            if px[0] == 0 and px[1] == 0 and px[2] == 0:
                pass
            else:
                img_back[loca_x + x, loca_y + y] = px
    return img_back


def img_mix(target_img, back_img):
    # 输入背景图片和目标图片，随机透视变换与旋转放缩，然后叠在背景图片上面，返回yolo数据集格式的列表与混合后图片
    # 主打一个强化学习
    # 正态分布，改变方差来改变目标图片被透视变化的扭曲程度
    one_normal = np.random.normal(loc=0.5, scale=0.1, size=(1, 4))
    a = one_normal[0, 0]
    d = one_normal[0, 1]
    b = one_normal[0, 2]
    c = one_normal[0, 3]
    # 处理随机出来的负数部分
    a = (abs(a) + a) / 2
    b = (abs(b) + b) / 2
    c = (abs(c) + c) / 2
    d = (abs(d) + d) / 2

    # 目标图片大小随机范围，不要大于背景图片（悲
    zoom_random_y = random.randint(200, 600)
    # 魔法数字(悲
    zoom_random_x = int(zoom_random_y * 1.69)
    target_img = cv2.resize(target_img, (zoom_random_x, zoom_random_y))
    cols, rows = zoom_random_x, zoom_random_y
    half_cols = cols / 2
    half_rows = rows / 2
    # row_r_cols = cols / rows(不知道有什么用，算了罢)
    # 计算四个需要放缩的点左边
    left_up_point = [int(half_cols - half_cols * a), int(half_rows - a * half_rows)]
    right_up_point = [int(half_cols + b * half_cols), int(half_rows - b * half_rows)]
    left_down_point = [int(half_cols - half_cols * c), int(half_rows + c * half_rows)]
    right_down_point = [int(half_cols + d * half_cols), int(half_rows + d * half_rows)]
    # 计算目标图片的尺寸
    x_sharp = int(half_cols * (1 - max(a, c)))
    y_sharp = int(half_rows * (1 - max(a, b)))

    left_up_point[0] = left_up_point[0] - x_sharp
    left_up_point[1] = left_up_point[1] - y_sharp
    right_down_point[0] = right_down_point[0] - x_sharp
    right_down_point[1] = right_down_point[1] - y_sharp
    left_down_point[0] = left_down_point[0] - x_sharp
    left_down_point[1] = left_down_point[1] - y_sharp
    right_up_point[0] = right_up_point[0] - x_sharp
    right_up_point[1] = right_up_point[1] - y_sharp
    targ_col = int(half_cols * max(b, d) + half_cols * max(a, c) * 1.3)
    targ_row = int(half_rows * max(a, b) + half_rows * max(c, d))
    # 原始图片四个角
    point1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    # print(cols)
    # print(rows)
    # 计算变换矩阵并应用
    point2 = np.float32([left_up_point, right_up_point, left_down_point, right_down_point])
    M = cv2.getPerspectiveTransform(point1, point2)
    dst = cv2.warpPerspective(target_img, M, (targ_col, targ_row))

    # 随机化图片位置并钳制其不超出图片范围
    position_x = random.randint(0, (1920 - cols))
    position_y = random.randint(0, (1080 - rows))
    back_img = cv2.resize(back_img, (1920, 1080))
    fina = mix_pic(dst, back_img, position_y, position_x)
    # back_img[0:af_cols, 0:af_rows] = dst
    # #cv2.imshow('fin', fina)
    # yolo列表
    yolo_format = [(position_x + targ_col / 2) / 1920, (position_y + targ_row / 2) / 1080, (targ_col / 1920) * 0.8,
                   (targ_row / 1080) * 0.8]

    return fina, yolo_format


# 重构不明白，开摆

def pre_make():
    global target_location
    # 这里有待改改
    src_folder = target_location
    dst_folder = "target/"

    # 获取源文件夹中所有子文件夹的路径
    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]
    shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)
    # 遍历每个子文件夹，并将其中的第一张图片复制到目标文件夹中
    for folder in subfolders:
        copied = False
        for file in os.listdir(folder):
            # 只处理jpg和png文件
            if file.endswith(".jpg") or file.endswith(".png"):
                # 构建目标文件路径，以子文件夹名称作为文件名前缀
                file_prefix = os.path.basename(folder)
                dst_file = os.path.join(dst_folder, file_prefix + "_" + file)
                # 复制文件并重命名
                shutil.copy(os.path.join(folder, file), dst_file)
                # 删除源文件
                os.remove(os.path.join(folder, file))
                # 设置标志位，表示已经复制了一张图片
                copied = True
                # 删除文件名中的数字
                dst_file_no_digit = re.sub(r"\t+", "", os.path.basename(dst_file))
                dst_file_no_digit = os.path.join(os.path.dirname(dst_file), dst_file_no_digit)
                os.rename(dst_file, dst_file_no_digit)
                # 仅复制第一张图片
                break
        # 如果没有复制任何图片，则输出提示信息
        if not copied:
            print("未找到任何图片：%s" % folder)
            return 114514
        # flag:辨别最后一轮输出


# 试图改成cmd命令行输入
def get_cmd_input():
    global huge_location, target_location, made_number
    parser = argparse.ArgumentParser(description='para transfer')
    # parser.add_argument('--para1', action='store_true', default=False, help='para1 -> bool type.')
    parser.add_argument('--back', type=str, default='back/', help='path of the back image')
    parser.add_argument('--target', type=str, default="images/", help='path of target image')
    parser.add_argument('--num', type=int, default=2, help='the number of each target made times of back images')
    args = parser.parse_args()
    huge_location = args.back
    target_location = args.target
    made_number = args.num


# 来点gui以获取用户输入，以后准备改成命令行版本（悲
def get_information():
    global huge_location, target_location, made_number
    yse_or_no = gui.confirm(text='make sure you read the readme.md"', title='setup running',
                            buttons=['yes', 'no'])

    if yse_or_no == 'no':
        quit()
    huge_location = gui.prompt(text='the path of background image', title='qwq', default='back/')
    target_location = gui.prompt(text='the path of target image', title='qwq', default='images/')
    made_number = gui.prompt(text='the number you want for each target image', title='qwq', default='10')
    made_number = int(made_number)


def read_target():
    global target_location, target_image_list, image_names_target
    # 读取目标的数量种类，存在列表里
    folder_path = target_location

    # 指定目录下的所有jpg和png文件
    img_files = glob.glob(folder_path + '*.jpg') + glob.glob(folder_path + '*.png')

    # 存储文件名的列表
    img_names = []

    # 存储图像的列表
    imgs = []

    # 遍历所有图像文件
    for file in img_files:
        # 读取图像
        img = cv2.imread(file)

        # 获取文件名并去掉扩展名
        # img_name = file.split('/')[-1].split('.')[0]
        img_name = os.path.splitext(os.path.basename(file))[0]
        # 添加到列表中
        imgs.append(img)
        img_names.append(img_name)
    target_image_list = imgs
    image_names_target = img_names


def make_obj():
    global image_names_target
    with open("obj.names", "w") as f:
        for c in image_names_target:
            # print(self.image_names_target)
            f.write(c + "\n")


def save_yolo_data(class_id, label, img, data_dir='labels/'):
    # Parse YOLO data

    boxes = []
    x, y, w, h = label
    boxes.append([class_id, x, y, w, h])

    # Save image
    img_name = str(time.time())
    img_save_path = 'imagess/' + img_name
    # print(img_save_path)
    img_save_path = img_save_path + '.png'
    cv2.imwrite(img_save_path, img)

    # Save YOLO data
    data_name = img_name + '.txt'
    data_save_path = os.path.join(data_dir, data_name)
    # print(data_save_path)
    with open(data_save_path, 'w') as f:
        for box in boxes:
            f.write('{} {} {} {} {}\n'.format(*box))


def read_back_image():
    global huge_location, target_image_list, made_number
    folder_path = huge_location
    i = -1
    for tg_img in target_image_list:
        i = i + 1
        print(i)
        print('len' + str(len(target_image_list)))
        for filename in os.listdir(folder_path):
            # print(filename)
            # print(os.listdir(folder_path))
            # 判断文件类型是否为jpg或png图片
            if filename.endswith('.jpg') or filename.endswith('.jfif') or filename.endswith('.jpg'):
                # print('2')
                # 读取图片
                img_path = os.path.join(folder_path, filename)
                # print(img_path)
                img = cv2.imread(img_path)

                # 调用处理函数
                j = 0
                while (j < made_number):
                    fina_img, yolo_num = img_mix(tg_img, img)
                    fina_img = cv2.resize(fina_img, (720, 480))
                    save_yolo_data(i, yolo_num, fina_img)
                    j = j + 1
            else:
                pass


# get_cmd_input()命令行版本寄了，回头写罢
get_information()
# maker maker
while pre_make() != 114514:
    read_target()
    make_obj()
    read_back_image()
