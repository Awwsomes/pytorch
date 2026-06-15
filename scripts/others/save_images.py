import cv2
import os
from tqdm import tqdm

root_path = r"D:\A_myData\dataset\juanZhou12"  # 存放视频的根目录
path_output = r"D:\A_myData\dataset\juanZhou13"
start_idx = 9142  # 请输入目前不存在的序号
save_per_frame = 10
video_extension = ('.mp4','.avi')

if not os.path.exists(path_output):
    os.mkdir(path_output)

list_video = os.listdir(root_path)

for idx,video in enumerate(list_video):
    if os.path.isdir(video):
        print("Note: {} not a video, skip.".format(video))
        continue
    else:
        video_name,extension = os.path.splitext(video)
        if not extension in video_extension:
            print("Note: {} not a video, skip.".format(video))
            continue

    path_video = os.path.join(root_path,video)
    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print("Open false")
        exit()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % save_per_frame == 0:
            output_path = os.path.join(path_output,"juanZhou{}.png".format(start_idx))
            # frame = cv2.resize(frame, (32,32))
            cv2.imwrite(output_path, frame)
            print("{}/{} Write juanZhou{}.png success.".format(idx+1,len(list_video),start_idx))
            start_idx = start_idx + 1
        frame_idx += 1
    cap.release()