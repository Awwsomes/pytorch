# import json
# import re
# import os
#
# txt_path = r"D:\A_myData\dataset\wuQiTou1\labels1"
#
# list_txt = os.listdir(txt_path)
#
# for txt in list_txt:
#     path_txt = os.path.join(txt_path,txt)
#     with open(path_txt,'r') as file:
#         lines = file.readlines()
#         len_line = len(lines)
#         if len_line > 1:
#             print("{} error.".format(path_txt))

# import torch
#
# # 在Windows系统上执行这段代码
# model = torch.load(r'D:\A_myData\Pytorch\yolov5-master\runs\train\exp25_wuQiTou\weights\best.pt', map_location='cpu',weights_only=False)
# torch.save(model, 'hou_li_v5s_1111.pt', _use_new_zipfile_serialization=False)

# import os
# import shutil
#
# input_path = r"D:\A_myData\dataset\juanZhou_gazebo6-cls"
# list_dir = os.listdir(input_path)
# for dir in list_dir:
#     path_dir = os.path.join(input_path,dir)
#     list_img = os.listdir(path_dir)
#     for img in list_img:
#         old_path = os.path.join(path_dir,img)
#         new_path = os.path.join(input_path,img)
#         shutil.copy(old_path,new_path)

# import os
# import shutil

# input_path = r"D:\A_myData\dataset\juanZhou_gazebo6-cls"
# list_dir = os.listdir(input_path)
# for dir in list_dir:
#     idx = dir.split("_")[1]
#     new_dir = str(int(idx) + 1)
#     new_path = os.path.join(input_path,new_dir)
#     old_path = os.path.join(input_path,dir)
#     os.rename(old_path,new_path)

# for root,dirs,files in os.walk("D:\A_myData\dataset\juanZhou_gazebo5-cls"):
#     print(f"{root}\n{dirs}\n{files}\n------------------")

# def copy_dirs_with_ori_name(root_dir,output_dir:str):
#     if not os.path.exists(root_dir):
#         print("根目录不存在！")
#         return -1
#     for root,dirs,_ in os.walk(root_dir):
#         if len(dirs) == 0:
#             rela_path = os.path.relpath(root,root_dir)
#             new_path = os.path.join(output_dir,rela_path)
#             os.makedirs(new_path,exist_ok=True)
#     print("目录拷贝完成.")
#     return 0
#
# copy_dirs_with_ori_name("D:\A_myData\dataset\juanZhou_gazebo5-cls","D:\A_myData\dataset\juanZhou_gazebo6")

import os
import shutil

# label_path = r"D:\A_myData\dataset\test_map50_cla_\labels"
# output_root_path = r"D:\A_myData\dataset\test_map50_cla_\labels_new"
# os.makedirs(output_root_path,exist_ok=True)
# list_label = os.listdir(label_path)
# for label in list_label:
#     label_name,label_ext = os.path.splitext(label)
#     # print(label_name,label_ext)
#     label_idx = label_name.split("_")[1]
#     # print(label_idx)
#     new_label = f"images_{label_idx}{label_ext}"
#     # print(new_label)
#     origin_path = os.path.join(label_path,label)
#     # print(origin_path)
#     output_path = os.path.join(output_root_path,new_label)
#     # print(output_path)
#     shutil.copy(origin_path,output_path)

# import re
#
# input_dir = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_26_output\images"
#
# list_img = os.listdir(input_dir)
# list_img.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else float('inf'))
# print(list_img)

import numpy as np

list_d = [1,2,3,4]
d = np.stack(list_d)

print(d)