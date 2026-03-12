import os
import json
import numpy as np
import re

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        points = np.zeros((4,2), dtype = np.float64)
        # print(data)
        for idx,point in enumerate(data["shapes"][0]["points"]):
            # print(point)
            points[idx,:] = point
    return points

def write_txt(output_txt_file,points,output_txt_path):
    for point in points:
        output_txt_file.write(str(point[0]) + " " + str(point[1]) + "\n")

def generate_corner_photo_txt(json_dir,output_path):
    json_list = os.listdir(json_dir)
    json_list = [x for x in json_list if os.path.isfile(os.path.join(json_dir,x))]
    json_list = [x for x in json_list if x.endswith(".json")]
    # 按时间戳排序
    json_list = sorted(json_list)
    print(json_list)
    with open(output_path, 'x') as txt_file:
        for json_file in json_list:
            points = read_json(os.path.join(json_dir,json_file))
            write_txt(txt_file,points,output_path)

if __name__ ==  "__main__":
    json_path = r"D:\A_myData\RC26-Vision\calibration\lidar_camera\260311\json"
    output_path = r"D:\A_myData\RC26-Vision\calibration\lidar_camera\260311\txt\corner_photo.txt"
    generate_corner_photo_txt(json_path,output_path)