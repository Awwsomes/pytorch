# import cv2
#
#
# # 3: 831,584 ; 691,457
# ## 6: 843,453 ; 718,360
# # 2: 374,582 ; 228,461
# # 5: 500,433 ; 397,356
# # 4: 130,463 ; 0,360
# # 7: 269,388 ; 173,305
# # 10: 366,343 ; 302,274
#
#
# # 读照片
# img1 = cv2.imread("D:\\A_myData\\dataset\\block\\img1.png")
# img2 = cv2.imread("D:\\A_myData\\dataset\\block\\img2.png")
# img3 = cv2.imread("D:\\A_myData\\dataset\\block\\img3.jpg")
# img4 = cv2.imread("D:\\A_myData\\dataset\\block\\img4.jpg")
# img5 = cv2.imread("D:\\A_myData\\dataset\\block\\img5.jpg")
# img6 = cv2.imread("D:\\A_myData\\dataset\\block\\img6.jpg")
#
# # 按照像素分割
# # img_cut1 = img1[457:584,691:831]
# # img_cut2 = img1[461:582,228:374]
# # img = img1[356:433,397:500]
# # img = img1[360:463,0:130]
# img_cut3_1 = img1[457:584,691:831]
# img_cut3_2 = img2[457:584,691:831]
# img_cut3_6 = img6[457:584,691:831]
# img_cut3 =
#
# # 展示
# cv2.imshow("img3",img3)
# cv2.imshow("img2",img2)
# cv2.imshow("img5",img5)
# cv2.imshow("img4",img4)
# cv2.waitKey(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt

### 两点
### 右下点；左上点
## r2启动区
# 2: 374,582 ; 228,461
# 3: 831,584 ; 691,457
# 4: 130,463 ; 0,360
# 5: 500,433 ; 397,356
# 7: 269,388 ; 173,305
# 10: 366,343 ; 302,274

## 梅花林2号位
# 4: 305,1070 ; 0,782
# 5: 1153,1080 ; 831,758
# 6: 1920,1080 ; 1660,820
# 7: 470,750 ; 223,547
# 8: 1072,745 ; 913,586   159
# 9: 1764,760 ; 1529,556
# 10: 621,525 ; 492，396   129
# 11: 1050,501 ; 916,367   134
# 12: 1461,525 ; 1284,348   177

## 梅花林2号位_新
#

## 2号位 相机面向1号位
# 1: 1100,1000 ; 800,730
# 4: 1920,1080 ; 1616,774   304

### 左上点+宽高
## r2启动区
# 2: 228,461 ; 146,121
# 3: 691,457 ; 140,127
# 4: 0,360 ; 130,103
# 5: 397,356 ; 103,77
# 7: 173,305 ; 96,83
# 10: 302,274 ; 64,69
## 2号位
# 4: 0,782 ; 305,288
# 5: 831,758 ; 322,322
# 6: 1660,820 ; 260,260
# 7: 223,547 ; 247,203
# 8: 913,586 ; 159,159
# 9: 1529,556 ; 235,204
# 10: 492,396 ; 129,129
# 11: 916,367 ; 134,134
# 12: 1284,348 ; 177,177
## 2号位 相机面向1号位
# 1: 800,730 ; 300,270
# 4: 1616,774 ; 304,304


def simple_crop_and_display(image_paths, bbox_coordinates):
    """
    简化版本：只显示拼接后的结果
    """

    # 读取图片
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)

    # 创建结果显示
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, bbox in enumerate(bbox_coordinates):
        x1, y1, x2, y2 = bbox

        # 裁剪并拼接
        cropped_images = [img[y1:y2, x1:x2] for img in images]
        concatenated = np.hstack(cropped_images)

        # 显示
        axes[i].imshow(concatenated)
        axes[i].set_title(f'位置 {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def simple_crop_two_images(image_paths, bbox_coordinates):
    """
    简化版本：只显示拼接后的结果（两张图片，9个框）
    """

    # 读取图片
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"错误: 无法读取图片 {path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)

    # 创建3x3的子图网格
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for i, bbox in enumerate(bbox_coordinates):
        x1, y1, x2, y2 = bbox

        # 裁剪并拼接
        cropped_images = []
        for img in images:
            # 确保坐标在图像范围内
            h, w = img.shape[:2]
            x1_clipped = max(0, min(x1, w - 1))
            y1_clipped = max(0, min(y1, h - 1))
            x2_clipped = max(0, min(x2, w))
            y2_clipped = max(0, min(y2, h))

            cropped = img[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
            cropped_images.append(cropped)

        concatenated = np.hstack(cropped_images)

        # 显示
        axes[i].imshow(concatenated)
        axes[i].set_title(f'位置 {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def crop_image_side(img_path,bbox):
    img = cv2.imread(img_path)
    frame1 = bbox[0]
    frame2 = bbox[1]
    img_cut1 = img[frame1[1]:frame1[3],frame1[0]:frame1[2]]
    img_cut2 = img[frame2[1]:frame2[3], frame2[0]:frame2[2]]

    cv2.imshow("img_cut1",img_cut1)
    cv2.imshow("img_cut4",img_cut2)
    if cv2.waitKey(0) == 'q':
        return

    # _, axes = plt.subplots(1, 2)
    # axes = axes.ravel()
    #
    # concatenated1 = np.hstack(img_cut1)
    # axes[0].imshow(concatenated1)
    # axes[0].set_title(f'位置 {1}')
    # axes[0].axis('off')
    # concatenated2 = np.hstack(img_cut2)
    # axes[1].imshow(concatenated2)
    # axes[1].set_title(f'位置 {2}')
    # axes[1].axis('off')
    #
    # plt.tight_layout()
    # plt.show()

# # r2启动区
# image_paths = ["D:\\A_myData\\dataset\\block\\img1.png", "D:\\A_myData\\dataset\\block\\img2.png", "D:\\A_myData\\dataset\\block\\img6.jpg"]
# bbox_coordinates = [
#     (228,461,374,582),
#     (691,457,831,584 ),
#     (0,360,130,463 ),
#     (397,356,500,433 ),
#     (173,305,269,388 ),
#     (302,274,366,343 )
# ]
# simple_crop_and_display(image_paths, bbox_coordinates)

# # 2号位 相机面向5号位
# image_paths = ["D:\\A_myData\\dataset\\block\\img4.jpg", "D:\\A_myData\\dataset\\block\\img5.jpg"]
# bbox_coordinates = [
#     (0,782,305,1070),
#     (831,758,1153,1080 ),
#     (1660,820,1920,1080 ),
#     (223,547,470,750  ),
#     (913,586,1072,745 ),
#     (1529,556,1764,760 ),
#     (492,396,621,525 ),
#     (916,367,1050,501 ),
#     (1284,348,1461,525 )
# ]
# simple_crop_two_images(image_paths, bbox_coordinates)

# 2号位 相机面向1号位
image_path = "D:\\A_myData\\dataset\\block\\img3.jpg"
bbox_coordinates = [
    (800,730,1100,1000),
    (1616,774,1920,1080)
]
crop_image_side(image_path,bbox_coordinates)