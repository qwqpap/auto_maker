import pyautogui as gui
import cv2
import numpy
import os
import time
import glob
import cv2
import numpy as np
import random
# import numba
from numba import jit
import shutil
import re
def pre_make():
    src_folder = "images/"
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
                dst_file_no_digit = re.sub(r"\d+", "", os.path.basename(dst_file))
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
    img_save_path = 'images/' + img_name
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

    one_normal = np.random.normal(loc=1.0, scale=0.2, size=(1, 2))
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
    dst = cv2.blur(dst, (3, 3))
    front_x, front_y = dst.shape[:2]

    back_img = cv2.resize(back_img, (1920, 1080))
    rand_x = random.randint(200, 400)
    rand_y = int(rand_x * (3/5))
    dst = cv2.resize(dst, (rand_x, rand_y))
    position_x = random.randint(0, (1920 - rand_x))
    position_y = random.randint(0, (1080 - rand_y))

    fina = mix_pic(dst, back_img, position_y, position_x)
    # back_img[0:af_cols, 0:af_rows] = dst
    #cv2.imshow('fin', fina)
    yolo_format = []
    #print(position_x)
    #print(position_y)
    #print(rand_x)
    #print(rand_y)
    yolo_format.append((position_x + rand_x / 2) / 1920)
    yolo_format.append((position_y + rand_y / 2) / 1080)
    yolo_format.append((rand_x / 1920)*0.8)
    yolo_format.append((rand_y / 1080)*0.8)
    #print(yolo_format)
    return fina, yolo_format
    #cv2.waitKey(0)


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
                if filename.endswith('.JPG') or filename.endswith('.jfif') or filename.endswith('.jpg'):
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
while pre_make() != 114514:
    AUTO.read_target()
    AUTO.make_obj()
    AUTO.read_back_image()