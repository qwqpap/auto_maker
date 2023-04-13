import os
import shutil
import time
import re
# 设置源文件夹路径和目标文件夹路径
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