import pyautogui as gui
import cv2
import numpy
import os


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
        self.target_location = gui.prompt(text='the path of background image', title='qwq', default='/data/target/')
        self.made_number = gui.prompt(text='the number you want for each target image', title='qwq', default='114514')

    def read_target(self):
        # 读取目标的数量种类，存在列表里
        image_list = os.listdir(self.huge_location)
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


    def make_them_together(self,background,targ):
        targ_image_size = targ.size
        back_image_size = background.size
        targ_x_size = targ_image_size[1]
        targ_y_size = targ_image_size[0]
        back_x_size = back_image_size[1]
        back_y_size = back_image_size[0]



AUTO = AutoMakerDataYolo()
AUTO.got_data()
