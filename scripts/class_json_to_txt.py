import os
import json
import yaml


def convert_labels(json_dir, yaml_path, save_txt_dir):
    """
    把json标签转换为YOLO的txt标签（四角点）
    :param json_dir:
    :param yaml_path:
    :param save_txt_dir:
    :return: none
    """
    # 加载类别名称映射
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    name_to_id = {v: str(k) for k, v in data['names'].items()}  # 确保ID是字符串类型

    # 确保保存目录存在
    os.makedirs(save_txt_dir, exist_ok=True)

    # 遍历所有JSON文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)

            try:
                # 读取JSON文件
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                # 处理shapes数组
                shapes = json_data.get('shapes', [])
                if not shapes:
                    print(f"警告：文件 {filename} 的shapes数组为空 - 跳过")
                    continue

                # 提取所有有效标签
                label_ids = []
                for shape in shapes:
                    label_name = shape.get('label')
                    if not label_name:
                        print(f"警告：文件 {filename} 中存在无label的shape - 跳过")
                        continue

                    label_id = name_to_id.get(label_name)
                    if label_id is None:
                        print(f"警告：类别 '{label_name}' 在配置文件中未找到 - 跳过")
                        continue

                    label_ids.append(label_id)

                # 写入txt文件
                if not label_ids:
                    print(f"警告：文件 {filename} 未提取到有效标签 - 跳过")
                    continue

                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(save_txt_dir, txt_filename)

                with open(txt_path, 'w') as f:
                    f.write('\n'.join(label_ids))

                print(f"成功处理：{filename} → {txt_filename} ({len(label_ids)}个标签)")

            except Exception as e:
                print(f"处理文件 {filename} 时出错：{str(e)}")

# 使用示例
json_directory = r"H:\pycharm\yolov5_proj4\yolov5\未分类\json"
yaml_config = r'I:\yolo\yolov11proj1\datasets\data.yaml'
output_directory = r'I:\yolo\yolo11proj_class1\dataset\txt'

convert_labels(json_directory, yaml_config, output_directory)