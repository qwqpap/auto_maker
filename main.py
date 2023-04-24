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
def pre_move():
    # 定义源目录和目标目录的路径
    source_dir = '/image'
    target_dir = '/images'

    # 检查目标目录是否存在，如果存在就清空它
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    # 创建目标目录
    os.makedirs(target_dir)

    # 递归遍历源目录
    for root, dirs, files in os.walk(source_dir):
        # 构造目标目录的路径
        target_root = os.path.join(target_dir, os.path.relpath(root, source_dir))
        # 如果目标目录不存在，就创建它
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        # 移动所有文件到目标目录中
        for file in files:
            source_path = os.path.join(root, file)
            target_path = os.path.join(target_root, file)
            shutil.move(source_path, target_path)
        # 移动当前目录到目标目录中
        if root != source_dir:
            shutil.move(root, target_root)

def pre_make():
    src_folder = "image/"
    dst_folder = "target/"



    # 获取源文件夹中所有子文件夹的路径
    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]
    shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)
    # 遍历每个子文件夹，并将其中的第一张图片复制到目标文件夹中Z
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
        #flags just do it

def save_yolo_data(class_id, label, img, data_dir='labels/'):
    # Parse YOLO data

    boxes = []
    x, y, w, h = label
    boxes.append([class_id, x, y, w, h])

    # Save image
    img_name = str(time.time())
    img_save_path = 'imagess/' + img_name
    #print(img_save_path)
    img_save_path = img_save_path + '.png'
    cv2.imwrite(img_save_path, img)

    # Save YOLO data
    data_name = img_name + '.txt'
    data_save_path = os.path.join(data_dir, data_name)
    #print(data_save_path)
    with open(data_save_path, 'w') as f:
        for box in boxes:
            f.write('{} {} {} {} {}\n'.format(*box))

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
    one_normal = np.random.normal(loc=0.5, scale=0.1, size=(1, 4))
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
    half_cols = cols / 2
    half_rows = rows / 2
    row_r_cols = cols / rows
    left_up_point = [int(half_cols - half_cols * a), int(half_rows - a * half_rows)]
    right_up_point = [int(half_cols + b * half_cols), int(half_rows - b * half_rows)]
    left_down_point = [int(half_cols - half_cols * c), int(half_rows + c * half_rows)]
    right_down_point = [int(half_cols + d * half_cols), int(half_rows + d * half_rows)]

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
    targ_col = int(half_cols * max(b, d) + half_cols * max(a, c))
    targ_row = int(half_rows * max(a, b) + half_rows * max(c, d))
    # print(left_up_point,left_down_point,right_up_point,right_down_point)
    point1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    # print(cols)
    # print(rows)
    point2 = np.float32([left_up_point, right_up_point, left_down_point, right_down_point])
    M = cv2.getPerspectiveTransform(point1, point2)
    dst = cv2.warpPerspective(target_img, M, (targ_col, targ_row))

    position_x = random.randint(0, (1920 - cols))
    position_y = random.randint(0, (1080 - rows))
    back_img = cv2.resize(back_img, (1920, 1080))
    fina = mix_pic(dst, back_img, position_y, position_x)
    # back_img[0:af_cols, 0:af_rows] = dst
    #cv2.imshow('fin', fina)
    yolo_format = []
    yolo_format.append((position_x + targ_col / 2) / 1920)
    yolo_format.append((position_y + targ_row / 2) / 1080)
    yolo_format.append((targ_col / 1920) * 0.8)
    yolo_format.append((targ_row / 1080) * 0.8)
    # print(position_x)
    # print(position_y)
    # print(rand_x)
    # print(rand_y)

    # print(yolo_format)
    #cv2.waitKey(0)
    return fina, yolo_format


class AutoMakerDataYolo:
    '''
    this is a class to automatic making the data of yolov5
    这是一个自动化生成yolov5数据集的库
    '''

    def __init__(self):
        self.all_kind_location = None
        self.image_names_target = None
        self.target_number = None
        self.target_name = None
        self.target_image_list = None
        self.made_number = None
        self.target_location = None
        self.huge_location = None
        self.num = 1
        self.yse_or_no = gui.confirm(text='make sure you read the readme.md"', title='setup runing',
                                     buttons=['yes', 'no'])

        if self.yse_or_no == 'no':
            quit()
        self.huge_location = gui.prompt(text='the path of background image', title='qwq', default='back/')
        self.target_location = gui.prompt(text='the path of target image', title='qwq', default='target/')
        self.made_number = gui.prompt(text='the number you want for each target image', title='qwq', default='10')
        self.made_number = int(self.made_number)




    def read_target(self):
        # 读取目标的数量种类，存在列表里
        folder_path = self.target_location

        # 指定目录下的所有jpg和png文件
        img_files = glob.glob(folder_path+'*.jpg') + glob.glob(folder_path+'*.png')

        # 存储文件名的列表
        img_names = []

        # 存储图像的列表
        imgs = []

        # 遍历所有图像文件
        for file in img_files:
            # 读取图像
            img = cv2.imread(file)

            # 获取文件名并去掉扩展名
            #img_name = file.split('/')[-1].split('.')[0]
            img_name = os.path.splitext(os.path.basename(file))[0]
            # 添加到列表中
            imgs.append(img)
            img_names.append(img_name)
        self.target_image_list = imgs
        self.image_names_target = img_names

    def read_back_image(self):
        folder_path = self.huge_location
        i = -1
        for tg_img in self.target_image_list:
            i = i + 1
            print(i)
            print('len'+str(len(self.target_image_list)))
            for filename in os.listdir(folder_path):
                # print(filename)
                # print(os.listdir(folder_path))
                # 判断文件类型是否为jpg或png图片
                if filename.endswith('.jpg') or filename.endswith('.jfif') or filename.endswith('.jpg'):
                    #print('2')
                    # 读取图片
                    img_path = os.path.join(folder_path, filename)
                    #print(img_path)
                    img = cv2.imread(img_path)

                    # 调用处理函数
                    j = 0
                    while(j < self.made_number):
                        fina_img, yolo_num = img_mix(tg_img, img)
                        # img_name = self.image_names_target[i]
                        fina_img = cv2.resize(fina_img,(720,480))
                        save_yolo_data(i, yolo_num, fina_img)
                        j = j + 1
                else:
                    pass
    def save_yolo_dataset_format(self,img_class, img_data, labels):
        save_path = 'data/'
        #print('1')
        # labels = self.image_names_target
        """
        Save data in YOLO dataset format with input labels, image class and img_data
        Args:
            labels (list): a list of bounding box labels, each label is a list of four values [x, y, w, h]
            img_class (int): class number of the image
            img_data (bytes): image data in binary format
            save_path (str): path to save yolo dataset format file
        """
        # calculate center coordinates, width and height values for each label
        yolo_labels = []
        # for label in labels:
        #print(labels)
        x, y, w, h = labels
        yolo_labels.append(f"{img_class} {x} {y} {w} {h}\n")

        # encode image data as JPEG format


    def make_obj(self):
        with open("obj.names", "w") as f:
            for c in self.image_names_target:
                #print(self.image_names_target)
                f.write(c + "\n")


AUTO = AutoMakerDataYolo()
#AUTO.got_data()
#pre_move()
while pre_make() != 114514:
    AUTO.read_target()
    AUTO.make_obj()
    AUTO.read_back_image()