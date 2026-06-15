import os
import cv2
import math

path_templates = r"C:\Users\tianc\Desktop\RC\juanZhou\Red"
path_output = r"C:\Users\tianc\Desktop\RC\juanZhou\Red_3"

list_templates = os.listdir(path_templates)

if not os.path.exists(path_output):
    os.mkdir(path_output)

# 遍历每一张
for template_name in list_templates:
    path_template = os.path.join(path_templates,template_name)
    template = cv2.imread(path_template)
    template = cv2.resize(template,(640,640))
    height, width, _ = template.shape
    # # 创建小根号2的图片
    # template_small = cv2.resize(template,None,fx=1/math.sqrt(2),fy=1/math.sqrt(2))
    # height_small,width_small,_ = template_small.shape
    center = [width//2,height//2]

    for idx,angle in enumerate(range(0,360,60)):
        # 获取旋转矩阵
        matrix = cv2.getRotationMatrix2D(center,angle,1/math.sqrt(2))

        # 旋转
        template_rotate = cv2.warpAffine(template,matrix,(width,height))

        cv2.imshow("rotate1",template_rotate)
        cv2.waitKey(10)

        img_name,img_extension = os.path.splitext(template_name)
        img_name += "_{}".format(idx)
        img_name += img_extension
        output_img_path = os.path.join(path_output,img_name)
        print(output_img_path)
        cv2.imwrite(output_img_path,template_rotate)