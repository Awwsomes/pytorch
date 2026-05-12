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
# import os
# import shutil
#
# import numpy as np
#
# # import torch
# #
# # # 在Windows系统上执行这段代码
# # model = torch.load(r'D:\A_myData\Pytorch\yolov5-master\runs\train\exp25_wuQiTou\weights\best.pt', map_location='cpu',weights_only=False)
# # torch.save(model, 'hou_li_v5s_1111.pt', _use_new_zipfile_serialization=False)
#
# # import os
# # import shutil
# #
# # input_path = r"D:\A_myData\dataset\juanZhou_gazebo6-cls"
# # list_dir = os.listdir(input_path)
# # for dir in list_dir:
# #     path_dir = os.path.join(input_path,dir)
# #     list_img = os.listdir(path_dir)
# #     for img in list_img:
# #         old_path = os.path.join(path_dir,img)
# #         new_path = os.path.join(input_path,img)
# #         shutil.copy(old_path,new_path)
#
# # import os
# # import shutil
#
# # input_path = r"D:\A_myData\dataset\juanZhou_gazebo6-cls"
# # list_dir = os.listdir(input_path)
# # for dir in list_dir:
# #     idx = dir.split("_")[1]
# #     new_dir = str(int(idx) + 1)
# #     new_path = os.path.join(input_path,new_dir)
# #     old_path = os.path.join(input_path,dir)
# #     os.rename(old_path,new_path)
#
# # for root,dirs,files in os.walk("D:\A_myData\dataset\juanZhou_gazebo5-cls"):
# #     print(f"{root}\n{dirs}\n{files}\n------------------")
#
# # def copy_dirs_with_ori_name(root_dir,output_dir:str):
# #     if not os.path.exists(root_dir):
# #         print("根目录不存在！")
# #         return -1
# #     for root,dirs,_ in os.walk(root_dir):
# #         if len(dirs) == 0:
# #             rela_path = os.path.relpath(root,root_dir)
# #             new_path = os.path.join(output_dir,rela_path)
# #             os.makedirs(new_path,exist_ok=True)
# #     print("目录拷贝完成.")
# #     return 0
# #
# # copy_dirs_with_ori_name("D:\A_myData\dataset\juanZhou_gazebo5-cls","D:\A_myData\dataset\juanZhou_gazebo6")
#
# # import os
# # import shutil
# #
# # # label_path = r"D:\A_myData\dataset\test_map50_cla_\labels"
# # # output_root_path = r"D:\A_myData\dataset\test_map50_cla_\labels_new"
# # # os.makedirs(output_root_path,exist_ok=True)
# # # list_label = os.listdir(label_path)
# # # for label in list_label:
# # #     label_name,label_ext = os.path.splitext(label)
# # #     # print(label_name,label_ext)
# # #     label_idx = label_name.split("_")[1]
# # #     # print(label_idx)
# # #     new_label = f"images_{label_idx}{label_ext}"
# # #     # print(new_label)
# # #     origin_path = os.path.join(label_path,label)
# # #     # print(origin_path)
# # #     output_path = os.path.join(output_root_path,new_label)
# # #     # print(output_path)
# # #     shutil.copy(origin_path,output_path)
# #
# # # import re
# # #
# # # input_dir = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_26_output\images"
# # #
# # # list_img = os.listdir(input_dir)
# # # list_img.sort(key=lambda f: int(re.findall(r'\d+', f)[0]) if re.findall(r'\d+', f) else float('inf'))
# # # print(list_img)
# #
# # import numpy as np
# #
# # list_d = [1,2,3,4]
# # d = np.stack(list_d)
# #
# # print(d)
#
# # import os
# # print(os.path.split("D:\A_myData\RC26-Vision\dataset\juanZhou_car1"))
#
# # import os
# # path = r"D:\Wechat File\xwechat_files\wxid_wlxqnttgybjx12_af55\msg\file\2026-03\2026_3_26\2026_3_26\4\imageRT\imagert1\0\image.png"
# # print(os.path.splitext(path)[0], os.path.splitext(path)[1])
#
# # from ultralytics import YOLO
# #
# # model = YOLO(r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\train\角点检测2_1000_260202\corner2_v5s_1000_260202.pt")
# # model.predict(
# #     source=r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_28\images\image_89.png",
# #     show=False,
# #     save_txt=True,
# #     save_conf=True,
# #     save=False,
# #     project=r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\detect"
# # )
# #
# # for i in range(0, -2, -1):
# #     print(i)
#
# # rvec = [1.1, 1.2, 1.3]
# # print(rvec)
# # json_content = (f"{{\n"
# #                 f"    \"rvec\": {rvec}\n"
# #                 f"}}")
# # print(json_content)
# # rvec1 = np.stack(rvec, axis=0).reshape(3,1)
# # print(rvec)
# # print(rvec1)
# # json_content = (f"{{\n"
# #                 f"    \"rvec\": {rvec}\n"
# #                 f"}}")
# # print(json_content)
#
# # for i in range(1,32):
# #     print(i)
# # list1 = [1,2,3,4,5,6,7,8,9]
# # [print("warn") if x not in range(1,33) else print(end="") for x in list1]
#
# # if list1 not in range(1,33):
# #     print("warn")
#
# # path = r"D:\A_myData\RC26-Vision\dataset\A_car\123"
# # os.mkdir(os.path.join(path, "1"))
#
# # import os
# # print(os.listdir(r"D:\A_myData\RC26-Vision\dataset\A_car\raw_data\2026_3_28\22\imageRT\imagert1"))
#
# # path = r"/home/awwsome/图片/摄像头/old"
# # new_path = r"/home/awwsome/图片"
# # list_img = os.listdir(path)
# # for img in list_img:
# #     img_name = img.split(".")[0]
# #     new_img_name = "d" + img_name + ".png"
# #     old_img_path = os.path.join(path, img)
# #     new_img_path = os.path.join(new_path, new_img_name)
# #     shutil.copy2(old_img_path, new_img_path)
#
# import cv2
# import numpy as np
# img_path = [r"C:\Users\tianc\Desktop\0000_new_1.png", r"C:\Users\tianc\Desktop\0001_new_1.png", r"C:\Users\tianc\Desktop\img.png"]
# imgs = [cv2.imread(x) for x in img_path]
# blank = np.zeros((1200, 600, 3), dtype=np.uint8)
# blank[:600, :] = imgs[0]
# blank[600:1200, :] = imgs[1]
# cv2.imwrite(r"D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\img.png", blank)

# from generate_car_datasets import generate_11class_dataset
# from config.car_dataset_config import config
# generate_11class_dataset(config.class_config.model_path,
#                          r"D:\A_myData\RC26-Vision\dataset\juanZhou_log\2026_1_9",
#                          r"D:\A_myData\RC26-Vision\dataset\juanZhou_log\2026_1_9_predict",
#                          config.class_config.start_idx,
#                          config.class_config.img_root_name,
#                          config.class_config.label_name_list,
#                          config.class_config.model_predict_output_dir)

# num = "0.11"
# num1 = int(num)
# print(num1)

# from generate_random_map_调车 import calculate_difference
#
# # difference_dict = calculate_difference([1,7,2,12,3,17,22,27,32,32,32,32], [32,32,8,13,5,18,23,28,32,4,32,1])
# difference_dict = calculate_difference([1, 19, 10, 32, 6, 6, 32, 15, 32, 31, 25, 32], [1, 19, 10, 32, 6, 6, 32, 15, 32, 31, 25, 32])
# print(difference_dict)

# from generate_car_datasets import generate_11class_dataset
# generate_11class_dataset(r"D:\A_myData\RC26-Vision\Pytorch\yolo11\runs\classify\卷轴分类4_4类\weights\best.pt",
#                          r"D:\A_myData\RC26-Vision\dataset\juanZhou_log\2026_1_12_new",
#                          r"D:\A_myData\RC26-Vision\dataset\juanZhou_log\2026_1_12_predict",
#                          0,
#                          "juanZhou_log2_",
#                          ["1", "2", "3", "4"],
#                          r"D:\A_myData\RC26-Vision\Pytorch\yolo11\runs\classify")

# import numpy as np
#
# # world_to_lidar = [[ 0.847164, -0.528611, 0.0536916, -1.91217],
# #  [0.531232,  0.840724, -0.104763, 0.0458141],
# # [0.0102391,  0.117274,  0.993047, -0.720668],
# #         [0,         0,         0,         1]]
#
# world_to_lidar =  [ [ 0.847164,    -0.53127, -0.00805017,    -1.86391],
# [   0.531232,    0.847202, -0.00643553 ,  -0.028872],
# [  0.0102391 , 0.00117545  ,  0.999947 ,  -0.838359],
#          [ 0    ,      0      ,     0     ,      1]]
#
# world_to_camera = [[-0.558015762354151, -0.8297200672348306, -0.013528451210104597, 0.1393677061599544],
# [0.00704477735943726, 0.011565483633255491, -0.9999083011458028, 1.288624180773572],
# [0.8298004459364142, -0.5580598978749123, -0.0006085290222784812, -1.5566266756168856],
# [0, 0, 0, 1]]
#
# output_matrix = world_to_camera @ np.linalg.inv(world_to_lidar)
#
# print(output_matrix)
#
# print(np.arange(1, 2))

# import os
# import shutil
#
# path1 = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_mix4"
# path2 = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_mix5"
#
# list1 = os.listdir(path1)
# list1 = [x for x in list1 if os.path.isdir(os.path.join(path1, x))]
# list1.sort(key=int)
# for dir1 in list1:
#     imgs = os.listdir(os.path.join(path1, dir1))
#     if dir1 == "1":
#         new_name = dir1
#     elif 1 < int(dir1) < 17:
#         new_name = "2"
#     elif 17 <= int(dir1) <= 31:
#         new_name = "3"
#     else:
#         new_name = "4"
#     print(f"{dir1} -> {new_name}")
#     new_path = os.path.join(path2, new_name)
#     os.makedirs(new_path, exist_ok=True)
#     for img in imgs:
#         old_img_path = os.path.join(path1, dir1, img)
#         new_img_path = os.path.join(new_path, img)
#         shutil.copy2(old_img_path, new_img_path)

# import os
# import shutil
#
# path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_log\2026_1_12"
# output_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_log\2026_1_12_new"
# os.makedirs(output_path, exist_ok=True)
# for root,dirs,files in os.walk(path):
#     # print(root, dirs, files)
#     if len(files) == 0:
#         continue
#     for file in files:
#         if not os.path.splitext(file)[1] in [".png", ".jpg", ".jpeg"]:
#             continue
#         elif file.find("hsv") != -1:
#             continue
#         old_path = os.path.join(root, file)
#         new_path = os.path.join(output_path, file)
#         shutil.copy2(old_path, new_path)

# import sys
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView,
#                              QGraphicsScene, QFileDialog)
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtCore import Qt
#
#
# class ImageViewer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.view = QGraphicsView()
#         self.scene = QGraphicsScene()
#         self.view.setScene(self.scene)
#         self.setCentralWidget(self.view)
#
#         # 启用滚轮缩放
#         self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # 拖拽平移
#         self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 光标中心缩放
#         self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
#
#     def wheelEvent(self, event):
#         # 滚轮缩放逻辑
#         zoom_factor = 1.15 if event.angleDelta().y() > 0 else 0.85
#         self.view.scale(zoom_factor, zoom_factor)
#
#     def open_image(self, path):
#         pixmap = QPixmap(path)
#         self.scene.clear()
#         self.scene.addPixmap(pixmap)
#         self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     viewer = ImageViewer()
#     viewer.show()
#     viewer.open_image(r'D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\img_example.png')  # 替换为你的图片路径
#     sys.exit(app.exec_())

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# # 读取并显示图片
# img = mpimg.imread(r'D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\img_example.png')
# plt.imshow(img)
# plt.axis('off')  # 隐藏坐标轴
# plt.show()  # 开启交互窗口，支持滚轮缩放、拖拽平移

# import cv2
# import numpy as np
#
#
# class ZoomableImage:
#     def __init__(self, image_path):
#         self.img = cv2.imread(image_path)
#         self.zoom_level = 1.0
#         self.window_name = "Zoomable Image"
#         cv2.namedWindow(self.window_name)
#         cv2.setMouseCallback(self.window_name, self.mouse_callback)
#         self.show_image()
#
#     def mouse_callback(self, event, x, y, flags, param):
#         if event == cv2.EVENT_MOUSEWHEEL:
#             # 滚轮事件：向上放大，向下缩小
#             if flags > 0:
#                 self.zoom_level *= 1.1
#             else:
#                 self.zoom_level *= 0.9
#             self.zoom_level = max(0.1, min(self.zoom_level, 10.0))  # 限制缩放范围
#             self.show_image()
#
#     def show_image(self):
#         # 缩放图像
#         h, w = self.img.shape[:2]
#         new_h, new_w = int(h * self.zoom_level), int(w * self.zoom_level)
#         zoomed_img = cv2.resize(self.img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#         cv2.imshow(self.window_name, zoomed_img)
#
#
# if __name__ == '__main__':
#     zoomer = ZoomableImage(r'D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\img_example.png')
#     while cv2.waitKey(0) & 0xFF == 27:  # ESC退出
#         cv2.destroyAllWindows()
#         break

# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
#
#
# class TkImageViewer:
#     def __init__(self, root):
#         self.root = root
#         self.canvas = tk.Canvas(root, bg='black')
#         self.canvas.pack(fill=tk.BOTH, expand=True)
#
#         self.scale = 1.0
#         self.orig_img = None
#         self.tk_img = None
#
#         # 绑定滚轮事件（Windows/Linux/Mac兼容）
#         self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows
#         self.canvas.bind("<Button-4>", self.on_zoom)  # Linux
#         self.canvas.bind("<Button-5>", self.on_zoom)  # Linux
#         self.canvas.bind("<B1-Motion>", self.on_drag)  # 拖拽平移
#         self.canvas.bind("<ButtonPress-1>", self.on_press)
#
#         self.offset_x = 0
#         self.offset_y = 0
#         self.last_x = 0
#         self.last_y = 0
#
#     def on_press(self, event):
#         self.last_x = event.x
#         self.last_y = event.y
#
#     def on_drag(self, event):
#         dx = event.x - self.last_x
#         dy = event.y - self.last_y
#         self.offset_x += dx
#         self.offset_y += dy
#         self.last_x = event.x
#         self.last_y = event.y
#         self.update_image()
#
#     def on_zoom(self, event):
#         # 滚轮缩放逻辑
#         delta = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
#         self.scale *= delta
#         self.scale = max(0.1, min(self.scale, 5.0))  # 限制缩放范围
#         self.update_image()
#
#     def open_image(self, path):
#         self.orig_img = Image.open(path)
#         self.update_image()
#
#     def update_image(self):
#         if self.orig_img:
#             # 缩放图像
#             w, h = self.orig_img.size
#             new_w = int(w * self.scale)
#             new_h = int(h * self.scale)
#             resized_img = self.orig_img.resize((new_w, new_h), Image.LANCZOS)
#             self.tk_img = ImageTk.PhotoImage(resized_img)
#
#             # 更新画布
#             self.canvas.delete("all")
#             self.canvas.create_image(self.offset_x, self.offset_y,
#                                      image=self.tk_img, anchor=tk.NW)
#
#
# if __name__ == '__main__':
#     root = tk.Tk()
#     root.title("Tkinter Image Viewer")
#     root.geometry("800x600")
#     viewer = TkImageViewer(root)
#     viewer.open_image(r'D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\img_example.png')  # 替换为你的图片路径
#     root.mainloop()

# from generate_random_map_调车 import map_12_to_4
# map1 = [1,2,3,9,10,16,17,18,19,28,29,30,31,32,32,32]
# print(map_12_to_4(map1))

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

# import os
# import json
#
#
# def check_labels_one_count_out_of_range(folder_path, min_allowed=0, max_allowed=float('inf'), recursive=True):
#     """
#     检查指定文件夹下所有JSON文件的labels列表中1的数量是否超出指定范围
#     当1的数量 < min_allowed 或 > max_allowed 时，打印提示信息
#
#     Args:
#         folder_path (str): 要检查的根文件夹路径
#         min_allowed (int): 允许的1的最小数量，默认为0（不限制下限）
#         max_allowed (int/float): 允许的1的最大数量，默认为无穷大（不限制上限）
#         recursive (bool): 是否递归检查子文件夹，默认为True
#     """
#     # 检查输入参数合法性
#     if min_allowed < 0:
#         print(f"错误：min_allowed不能为负数，当前值: {min_allowed}")
#         return
#
#     if max_allowed < min_allowed:
#         print(f"错误：max_allowed({max_allowed})不能小于min_allowed({min_allowed})")
#         return
#
#     # 检查文件夹是否存在
#     if not os.path.isdir(folder_path):
#         print(f"错误：文件夹 '{folder_path}' 不存在或不是一个有效的目录")
#         return
#
#     # 统计信息
#     total_files = 0
#     json_files = 0
#     out_of_range_files = 0
#
#     # 构建范围描述文本
#     range_desc = ""
#     if min_allowed == 0 and max_allowed == float('inf'):
#         range_desc = "无限制（所有文件都会被打印）"
#     elif min_allowed == 0:
#         range_desc = f"1的数量 > {max_allowed}"
#     elif max_allowed == float('inf'):
#         range_desc = f"1的数量 < {min_allowed}"
#     else:
#         range_desc = f"1的数量不在 [{min_allowed}, {max_allowed}] 范围内"
#
#     print(f"开始检查文件夹: {folder_path}")
#     print(f"触发条件: {range_desc}")
#     print(f"检查模式: {'递归遍历所有子文件夹' if recursive else '仅检查当前文件夹'}")
#     print("-" * 70)
#
#     # 遍历文件夹（支持递归）
#     if recursive:
#         walker = os.walk(folder_path)
#     else:
#         walker = [(folder_path, [], os.listdir(folder_path))]
#
#     for root, dirs, files in walker:
#         for filename in files:
#             total_files += 1
#             file_path = os.path.join(root, filename)
#
#             # 只处理.json文件
#             if not filename.lower().endswith('.json'):
#                 continue
#
#             json_files += 1
#
#             try:
#                 # 读取并解析JSON文件
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#
#                 # 检查是否存在labels键
#                 if 'labels' not in data:
#                     print(f"⚠️  文件 '{file_path}' 中没有找到 'labels' 键")
#                     continue
#
#                 labels = data['labels']
#
#                 # 检查labels是否是列表
#                 if not isinstance(labels, list):
#                     print(f"⚠️  文件 '{file_path}' 中的 'labels' 不是一个列表，类型为: {type(labels).__name__}")
#                     continue
#
#                 # 检查列表是否为空
#                 if len(labels) == 0:
#                     print(f"⚠️  文件 '{file_path}' 中的 'labels' 是一个空列表")
#                     continue
#
#                 # 统计列表中1的数量（支持整数1和浮点数1.0）
#                 one_count = sum(1 for item in labels if item == 1)
#
#                 # 检查是否超出范围
#                 is_out_of_range = False
#                 reason = ""
#
#                 if one_count < min_allowed:
#                     is_out_of_range = True
#                     reason = f"低于允许的最小值 {min_allowed}"
#                 elif one_count > max_allowed:
#                     is_out_of_range = True
#                     reason = f"高于允许的最大值 {max_allowed}"
#
#                 if is_out_of_range:
#                     out_of_range_files += 1
#                     print(f"❌ 文件 '{file_path}'")
#                     print(f"   labels总长度: {len(labels)}, 1的数量: {one_count}")
#                     print(f"   原因: {reason}")
#                     print()
#
#             except json.JSONDecodeError:
#                 print(f"❌ 文件 '{file_path}' 不是有效的JSON格式，解析失败")
#             except PermissionError:
#                 print(f"❌ 没有权限读取文件 '{file_path}'")
#             except Exception as e:
#                 print(f"❌ 处理文件 '{file_path}' 时发生未知错误: {str(e)}")
#
#     # 打印最终汇总信息
#     print("=" * 70)
#     print("检查完成！汇总信息:")
#     print(f"总文件数: {total_files}")
#     print(f"JSON文件数: {json_files}")
#     print(f"超出范围的文件数: {out_of_range_files}")
#     print(f"允许的范围: [{min_allowed}, {'∞' if max_allowed == float('inf') else max_allowed}]")
#     print("=" * 70)
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 替换为你的根文件夹路径
#     target_folder = "/home/awwsome/datasets/car_red/test_datas"
#     min_allowed = 6
#     max_allowed = 8
#
#     # 示例1：检查1的数量不在范围内的文件（最常用）
#     print(f"=== 检查1的数量不在[{min_allowed},{max_allowed}]范围内的文件 ===")
#     check_labels_one_count_out_of_range(target_folder, min_allowed=min_allowed, max_allowed=max_allowed)

    # 示例2：只检查上限（1的数量 > 10 的文件）
    # print("\n=== 检查1的数量 > 10 的文件 ===")
    # check_labels_one_count_out_of_range(target_folder, max_allowed=10)

    # 示例3：只检查下限（1的数量 < 3 的文件）
    # print("\n=== 检查1的数量 < 3 的文件 ===")
    # check_labels_one_count_out_of_range(target_folder, min_allowed=3)

    # 示例4：不递归检查
    # check_labels_one_count_out_of_range(target_folder, min_allowed=2, max_allowed=5, recursive=False)

import os
import shutil
import numpy as np
from PIL import Image


def check_subfolders_for_near_black_images(
        root_folder,
        min_near_black_images=2,
        black_threshold=0.1,
        delete_problematic=False
):
    """
    检查根文件夹下所有直接子文件夹中的图片，统计接近全黑图片的数量
    接近全黑定义：非0像素占总像素的比例 < black_threshold
    当子文件夹中接近全黑图片数量≥min_near_black_images时，可选择自动删除该子文件夹

    Args:
        root_folder (str): 根文件夹路径
        min_near_black_images (int): 触发提示/删除的最小接近全黑图片数量，默认为2
        black_threshold (float): 接近全黑的阈值，非0像素占比小于此值即判定为接近全黑，默认为0.1（10%）
        delete_problematic (bool): 是否自动删除有问题的子文件夹，默认为False（安全模式）
    """
    # 检查输入参数合法性
    if not (0 < black_threshold < 1):
        print(f"错误：black_threshold必须在(0, 1)范围内，当前值: {black_threshold}")
        return

    if min_near_black_images < 1:
        print(f"错误：min_near_black_images必须大于等于1，当前值: {min_near_black_images}")
        return

    # 检查根文件夹是否存在
    if not os.path.isdir(root_folder):
        print(f"错误：根文件夹 '{root_folder}' 不存在或不是有效的目录")
        return

    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # 统计信息
    total_subfolders = 0
    problematic_subfolders = []
    deleted_subfolders = []
    total_images_checked = 0
    total_near_black_images = 0

    print(f"开始检查文件夹: {root_folder}")
    print(f"接近全黑定义: 非0像素占比 < {black_threshold * 100}%")
    print(f"触发条件: 子文件夹内接近全黑图片数量 ≥ {min_near_black_images}")
    if delete_problematic:
        print("⚠️  警告：删除模式已开启，符合条件的子文件夹将被永久删除！")
    print("-" * 70)

    # 获取所有直接子文件夹
    subfolders = []
    for entry in os.listdir(root_folder):
        entry_path = os.path.join(root_folder, entry)
        if os.path.isdir(entry_path):
            subfolders.append(entry_path)

    if not subfolders:
        print("警告：根文件夹下没有找到任何子文件夹")
        return

    # 遍历每个子文件夹
    for subfolder in subfolders:
        total_subfolders += 1
        near_black_count = 0
        image_count = 0

        # 遍历子文件夹中的所有文件
        for filename in os.listdir(subfolder):
            file_path = os.path.join(subfolder, filename)

            # 只处理图片文件
            if not filename.lower().endswith(image_extensions):
                continue

            image_count += 1
            total_images_checked += 1

            try:
                # 打开图片并转换为灰度图
                with Image.open(file_path) as img:
                    gray_img = img.convert('L')
                    # 转换为numpy数组进行高效计算
                    gray_array = np.array(gray_img)

                    # 计算总像素数和非0像素数
                    total_pixels = gray_array.size
                    non_zero_pixels = np.count_nonzero(gray_array)
                    non_zero_ratio = non_zero_pixels / total_pixels

                    # 判断是否为接近全黑图片
                    if non_zero_ratio < black_threshold:
                        near_black_count += 1
                        total_near_black_images += 1
                        # 可选：打印每个接近全黑图片的详细信息
                        # print(f"📸 接近全黑图片: {file_path}")
                        # print(f"   非0像素占比: {non_zero_ratio:.4f} ({non_zero_pixels}/{total_pixels})")

            except Exception as e:
                print(f"⚠️  无法处理图片 '{file_path}': {str(e)}")
                continue

        # 检查是否达到触发条件
        if near_black_count >= min_near_black_images:
            problematic_subfolders.append(subfolder)
            print(f"❌ 问题文件夹: {subfolder}")
            print(f"   图片总数: {image_count}, 接近全黑图片数: {near_black_count}")

            # 如果开启了删除模式，则删除该子文件夹
            if delete_problematic:
                try:
                    shutil.rmtree(subfolder)
                    deleted_subfolders.append(subfolder)
                    print(f"   ✅ 已成功删除该文件夹")
                except Exception as e:
                    print(f"   ❌ 删除失败: {str(e)}")

            print()

    # 打印最终汇总信息
    print("=" * 70)
    print("检查完成！汇总信息:")
    print(f"总子文件夹数: {total_subfolders}")
    print(f"有问题的子文件夹数: {len(problematic_subfolders)}")
    if delete_problematic:
        print(f"已成功删除的子文件夹数: {len(deleted_subfolders)}")
    print(f"总共检查图片数: {total_images_checked}")
    print(f"总共发现接近全黑图片数: {total_near_black_images}")
    print(f"使用的接近全黑阈值: {black_threshold * 100}%")

    if problematic_subfolders:
        print("\n有问题的子文件夹列表:")
        for i, folder in enumerate(problematic_subfolders, 1):
            status = "已删除" if folder in deleted_subfolders else "保留"
            print(f"  {i}. {folder} [{status}]")
    else:
        print(f"\n✅ 所有子文件夹都符合要求，没有发现接近全黑图片数量≥{min_near_black_images}的情况")
    print("=" * 70)


# 使用示例
if __name__ == "__main__":
    # 替换为你的根文件夹路径
    target_root_folder = "/home/awwsome/datasets/car_blue/roi_images"

    # # 1. 安全模式（默认）：只检查不删除，使用默认阈值10%
    # print("=== 安全模式：只检查不删除 ===")
    # check_subfolders_for_near_black_images(target_root_folder)

    # 2. 删除模式：检查并删除有问题的子文件夹
    # 注意：请先在安全模式下确认结果无误后再使用删除模式！
    print("\n=== 删除模式：检查并删除有问题的子文件夹 ===")
    check_subfolders_for_near_black_images(
        target_root_folder,
        min_near_black_images=2,
        black_threshold=0.1,
        delete_problematic=True
    )

    # 3. 自定义阈值：非0像素占比<5%才判定为接近全黑
    # check_subfolders_for_near_black_images(
    #     target_root_folder,
    #     min_near_black_images=2,
    #     black_threshold=0.05,
    #     delete_problematic=False
    # )