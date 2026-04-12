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
import os

import numpy as np

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

# import os
# import shutil
#
# # label_path = r"D:\A_myData\dataset\test_map50_cla_\labels"
# # output_root_path = r"D:\A_myData\dataset\test_map50_cla_\labels_new"
# # os.makedirs(output_root_path,exist_ok=True)
# # list_label = os.listdir(label_path)
# # for label in list_label:
# #     label_name,label_ext = os.path.splitext(label)
# #     # print(label_name,label_ext)
# #     label_idx = label_name.split("_")[1]
# #     # print(label_idx)
# #     new_label = f"images_{label_idx}{label_ext}"
# #     # print(new_label)
# #     origin_path = os.path.join(label_path,label)
# #     # print(origin_path)
# #     output_path = os.path.join(output_root_path,new_label)
# #     # print(output_path)
# #     shutil.copy(origin_path,output_path)
#
# # import re
# #
# # input_dir = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_26_output\images"
# #
# # list_img = os.listdir(input_dir)
# # list_img.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else float('inf'))
# # print(list_img)
#
# import numpy as np
#
# list_d = [1,2,3,4]
# d = np.stack(list_d)
#
# print(d)

# import os
# print(os.path.split("D:\A_myData\RC26-Vision\dataset\juanZhou_car1"))

# import os
# path = r"D:\Wechat File\xwechat_files\wxid_wlxqnttgybjx12_af55\msg\file\2026-03\2026_3_26\2026_3_26\4\imageRT\imagert1\0\image.png"
# print(os.path.splitext(path)[0], os.path.splitext(path)[1])

# from ultralytics import YOLO
#
# model = YOLO(r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\train\角点检测2_1000_260202\corner2_v5s_1000_260202.pt")
# model.predict(
#     source=r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_28\images\image_89.png",
#     show=False,
#     save_txt=True,
#     save_conf=True,
#     save=False,
#     project=r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\detect"
# )
#
# for i in range(0, -2, -1):
#     print(i)

# rvec = [1.1, 1.2, 1.3]
# print(rvec)
# json_content = (f"{{\n"
#                 f"    \"rvec\": {rvec}\n"
#                 f"}}")
# print(json_content)
# rvec1 = np.stack(rvec, axis=0).reshape(3,1)
# print(rvec)
# print(rvec1)
# json_content = (f"{{\n"
#                 f"    \"rvec\": {rvec}\n"
#                 f"}}")
# print(json_content)

# for i in range(1,32):
#     print(i)
# list1 = [1,2,3,4,5,6,7,8,9]
# [print("warn") if x not in range(1,33) else print(end="") for x in list1]

# if list1 not in range(1,33):
#     print("warn")

# path = r"D:\A_myData\RC26-Vision\dataset\A_car\123"
# os.mkdir(os.path.join(path, "1"))

# import os
# print(os.listdir(r"D:\A_myData\RC26-Vision\dataset\A_car\raw_data\2026_3_28\22\imageRT\imagert1"))

# import os
# import numpy as np
# np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
# import cv2
# from scipy.spatial.transform import Rotation as R
#
# root_path = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\imageRT"
# for i in range(1,21,2):
#     imagert1_path = os.path.join(root_path, f"imagert{i}")
#     imagert2_path = os.path.join(root_path, f"imagert{i+1}")
#     for idx in range(2):
#         # print(f"idx: {i+idx}")
#         before_opt_path = os.path.join(imagert1_path, f"{idx}", "rt.txt")
#         after_opt_path = os.path.join(imagert2_path, f"{idx}", "rt.txt")
#         # print(before_opt_path)
#         # print(after_opt_path)
#         with open(before_opt_path, 'r') as before_file, \
#              open(after_opt_path, 'r') as after_file:
#             before_opt_rvec = np.array(before_file.readline().strip().split(", "))
#             before_opt_tvec = np.array(before_file.readline().strip().split(", "))
#             after_opt_rvec = np.array(after_file.readline().strip().split(", "))
#             after_opt_tvec = np.array(after_file.readline().strip().split(", "))
#
#         # before_R, _ = cv2.Rodrigues(before_opt_rvec)
#         rot = R.from_rotvec(before_opt_rvec)
#         before_R = rot.as_matrix()
#         before_RT = np.eye(4, dtype=np.float64)
#         before_RT[:3, :3] = before_R
#         before_RT[:3, 3] = before_opt_tvec
#         # print(before_RT)
#
#         rot = R.from_rotvec(after_opt_rvec)
#         after_R = rot.as_matrix()
#         after_RT = np.eye(4, dtype=np.float64)
#         after_RT[:3, :3] = after_R
#         after_RT[:3, 3] = after_opt_tvec
#         # print(after_RT)
#
#         #optimize_matrix = after_RT * np.linalg.inv(before_RT)
#         optimize_matrix = after_RT @ np.linalg.inv(before_RT)
#         # print(optimize_matrix)
#         # print(before_opt_rvec)
#         # print(after_opt_rvec)
#
#         print(f"第{i + idx}组：\n"
#               f"{optimize_matrix.astype(np.float64)}")