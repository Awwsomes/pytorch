import os
import json
import warnings

def json_rectangle_to_txt_standard_yolo(input_dir:str,output_dir:str,list_labels:list):
    """
    将输入的json（矩形框，有左上右下两点）转换为标准yolo格式的txt

    json格式：
        "points": [
            [
              1714.1176470588236,
              655.8823529411765
            ],
            [
              1738.8235294117646,
              675.8823529411765
            ]
          ]
    txt格式：
        x_center, y_center, width, height

    :param input_dir: 输入json文件夹
    :param output_dir: 输出txt文件夹
    :param list_labels: 类别映射列表
    :return: none
    """

    # 遍历json文件夹
    list_json = os.listdir(input_dir)
    list_json = [x for x in list_json if os.path.isfile(os.path.join(input_dir,x))]
    list_json = [x for x in list_json if x.endswith(".json")]

    os.makedirs(output_dir,exist_ok=True)

    for json_name in list_json:
        # 拼接路径
        origin_path = os.path.join(input_dir,json_name)
        output_name = os.path.splitext(json_name)[0] + ".txt"
        output_path = os.path.join(output_dir,output_name)

        # 打开输入输出文件（输出文件不能提前存在，x模式会报错）
        with open(origin_path,'r') as json_file, \
             open(output_path,'x') as txt_file:
            json_data = json.load(json_file)

            # 遍历每个框
            shapes = json_data["shapes"]
            for shape in shapes:
                # 计算框中心点坐标，宽高
                points = shape["points"]
                width = points[1][0] - points[0][0]
                height = points[1][1] - points[0][1]
                x_center = points[0][0] + width / 2
                y_center = points[0][1] + height / 2

                # 归一化
                width = width / 1920
                height = height / 1080
                x_center = x_center / 1920
                y_center = y_center / 1080

                # 判断json中的类别是否存在于输入的类别列表
                label = shape["label"]
                if label in list_labels:
                    label_idx = list_labels.index(label)
                    output_line = f"{str(label_idx)} {str(x_center)} {str(y_center)} {str(width)} {str(height)}\n"
                    txt_file.write(output_line)
                else:
                    warnings.warn(f"{json_name} 's {label} not in labels_list, skip...")
                    continue
            print(f"Write {output_name}.")

if __name__ == "__main__":
    input_path = r"D:\A_myData\dataset\corner3\jsons"
    output_path = r"D:\A_myData\dataset\corner3\labels"
    json_rectangle_to_txt_standard_yolo(input_path,output_path,["corner"])