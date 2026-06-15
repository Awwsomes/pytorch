import os
import cv2
import numpy
import numpy as np
from tqdm import tqdm

# 输入图片和txt标签路径
path_imgs = r"D:\A_myData\dataset\juanZhou3\train\images"
path_txts = r"D:\A_myData\dataset\juanZhou3\train\labels"
path_masks = r"D:\A_myData\dataset\juanZhou3\train\masks"

if not os.path.exists(path_masks):
    os.mkdir(path_masks)

# 遍历图片，读取同名txt标签（空处理），读取行
list_imgs = os.listdir(path_imgs)
list_txts = os.listdir(path_txts)
for img in tqdm(list_imgs):
    # img_name = img.split('.')[0]
    # img_extension_name = img.split('.')[1]
    img_name,img_extension_name = os.path.splitext(img)
    txt = img_name + ".txt"
    if not txt in list_txts:
        print("warning: {} not exists,skip.".format(txt))
        continue
    path_txt = os.path.join(path_txts,txt)

    path_img = os.path.join(path_imgs, img)
    image = cv2.imread(path_img)
    height_img, width_img, _ = image.shape
    mask = np.zeros((height_img, width_img), np.uint8)

    points = numpy.zeros([4,2],dtype=np.float32)
    with open(path_txt,'r') as txt_file:
        for line in txt_file:
            part = line.split(' ')
            part = [float(x) for x in part]
            length_part = len(part)
            # 矩形框（两点）
            if length_part == 5:
                point_left_up = [part[1],part[2]]
                point_right_down = [part[3],part[4]]
                width = point_right_down[0] - point_left_up[0]
                height = point_right_down[1] - point_left_up[1]
                point_left_down = [point_left_up[0],point_left_up[1] + height]
                point_right_up = [point_right_down[0],point_right_down[1] - height]
                points[0,:] = point_left_up
                points[1,:] = point_left_down
                points[2,:] = point_right_down
                points[3,:] = point_right_up
            # 多边形（四点）
            elif length_part == 9:
                points[0,:] = [part[1],part[2]]
                points[1,:] = [part[3],part[4]]
                points[2, :] = [part[5], part[6]]
                points[3, :] = [part[7], part[8]]

            # 坐标反归一化
            points[:,0] = points[:,0]*width_img
            points[:,1] = points[:,1]*height_img

            # 找最大外接圆，圆心和半径
            contour = points.reshape((-1,1,2))
            # print(contour)
            circle_point,r = cv2.minEnclosingCircle(contour)
            circle_point = (int(circle_point[0]),int(circle_point[1]))
            # print(circle_point)
            # print(int(r))

            # 创建同大小单通道图，画圆
            cv2.circle(mask,circle_point,int(r),255,-1)

            # img = np.zeros((400, 400, 3), dtype=np.uint8)
            #
            # # 设置圆心、半径、颜色、线条粗细
            # center = (200, 200)  # 圆心坐标 (x, y)
            # radius = 80  # 半径
            # color = (0, 0, 255)  # 红色 (BGR 格式)
            # thickness = 2  # 线条宽度，-1 表示填充圆
            #
            # # 在图像上画圆
            # cv2.circle(img, center, radius, color, thickness)

    # 存照片，名字加 _mask
    mask_name = img_name + "_mask" + img_extension_name
    path_mask = os.path.join(path_masks,mask_name)
    cv2.imwrite(path_mask,mask)