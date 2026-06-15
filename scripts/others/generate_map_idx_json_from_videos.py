import cv2
import os
import json

def generate_map_idx_json_from_video(path_video,path_output,output_img_name,map_idx_list: list,start_idx):
        cap = cv2.VideoCapture(path_video)
        if not cap.isOpened():
            print("open false")
            exit()

        i = 0
        idx = start_idx
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % 30 == 1 and i > 30:
                os.makedirs(os.path.join(path_output,"images"),exist_ok=True)
                output_img_path = os.path.join(path_output,"images",f"{output_img_name}{idx}.png")
                # frame = cv2.resize(frame, (32,32))
                cv2.imwrite(output_img_path, frame)

                os.makedirs(os.path.join(path_output, "jsons"), exist_ok=True)
                output_json_path = os.path.join(path_output,"jsons",f"{output_img_name}{idx}.json")
                json_data = {"labels":map_idx_list}
                with open(output_json_path,'w', encoding="utf-8") as json_file:
                    # json.dump参数说明：
                    # - data：要写入的Python数据
                    # - f：文件对象
                    # - ensure_ascii=False：保留中文等非ASCII字符（否则会转成\u编码）
                    # - indent=4：格式化输出，缩进4个空格（可读性更好）
                    json.dump(json_data, json_file, ensure_ascii=False)

                print(f"{path_video} save {idx - start_idx + 1}")
                idx += 1
            i = i + 1
        cap.release()

if __name__ == "__main__":
    # ## 单纯读取文件夹下所有视频，全部保存到一个文件夹中，一直编号
    # path_videos = r""
    # path_output = r""
    # output_img_name = ""
    # video_extension = [".mp4", ".avi", ".mov", ".wmv", ".flv"]
    #
    # list_video = os.listdir(path_videos)
    # list_video = [x for x in list_video if os.path.isfile(os.path.join(path_videos, x))]
    # list_video = [x for x in list_video if os.path.splitext(x)[1] in video_extension]
    #
    # if not os.path.exists(path_output):
    #     os.mkdir(path_output)
    #
    # for j, video in enumerate(list_video):
    #     path_video = os.path.join(path_videos, video)
    path_video = r"D:\A_myData\RC26-Vision\dataset\z_video\corner3\corner10.mp4"
    path_output = r"D:\A_myData\RC26-Vision\dataset\corner6"
    output_img_name = "label_"
    map = [1,1,0,0,1,1,1,0,0,1,1,1]
    generate_map_idx_json_from_video(path_video,path_output,output_img_name,map,5011)