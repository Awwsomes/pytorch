import cv2
import os

i = 0
idx = 0

origin_idx=idx

## 单纯读取文件夹下所有视频，全部保存到一个文件夹中，一直编号
path_videos = r"D:\A_myData\RC26-Vision\dataset\z_video\corner4"
path_output = r"D:\A_myData\RC26-Vision\dataset\corner9"
output_img_name = "corner9_"
video_extension = [".mp4",".avi",".mov",".wmv",".flv"]

list_video = os.listdir(path_videos)
list_video = [x for x in list_video if os.path.isfile(os.path.join(path_videos,x))]
list_video = [x for x in list_video if os.path.splitext(x)[1] in video_extension]

if not os.path.exists(path_output):
    os.mkdir(path_output)

for j,video in enumerate(list_video):
    path_video = os.path.join(path_videos,video)
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print("open false")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % 5 == 1 and i > 30:
            img_name = output_img_name + "{}.png".format(idx)
            output_path = os.path.join(path_output,img_name)
            # frame = cv2.resize(frame, (32,32))
            cv2.imwrite(output_path, frame)
            print(f"{j+1}/{len(list_video)} save {idx - origin_idx + 1}")
            idx += 1
        i = i + 1
    cap.release()

# ## 这版，能根据视频的名字（只有1.mp4这种）读取类别，分文件夹存放照片，但是序号一直向下编
# path_videos = r"D:\\A_myData\\dataset\\juanZhou16"
# path_output = r"D:\A_myData\dataset\juanZhou17"
#
# if not os.path.exists(path_output):
#     os.mkdir(path_output)
#
# list_video = os.listdir(path_videos)
# # print(list_video)
#
# for j,video in enumerate(list_video):
#     path_video = os.path.join(path_videos,video)
#     cap = cv2.VideoCapture(path_video)
#     if not cap.isOpened():
#         print("open false")
#         exit()
#
#     video_idx, _ = os.path.splitext(video)
#     path_output1 = os.path.join(path_output,video_idx)
#     os.mkdir(path_output1)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if i % 10 == 1 :
#             output_path = os.path.join(path_output1,"juanZhou{}.png".format(idx))
#             # frame = cv2.resize(frame, (32,32))
#             cv2.imwrite(output_path, frame)
#             print("保存图片{}".format(idx))
#             idx += 1
#         i = i + 1
#     cap.release()