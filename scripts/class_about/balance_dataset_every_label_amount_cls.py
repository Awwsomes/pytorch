import os
import shutil
import random
from tqdm import tqdm

import sys
# 把上两级目录（也就是和 rename_data.py 同级的项目根目录）加入 Python 搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from countLabel_cls import count_label_cls
from rename_data import copy_dirs

def balance_amount(root_path:str) -> dict[str, list]:
    """
    平衡分类数据集各类别的数量，以最少的数量为基准，最多向上浮动20张，随机抽取
    :param root_path: 分类数据集根目录
    :return: 字典[“类别名”：该类别被选中的图片路径列表]
    """
    # 获取最小值
    output_dict = count_label_cls(root_path)
    label_amount = output_dict["label_amount"]
    min_label_amount = output_dict["min_amount"]["amount"]
    # 遍历类别
    output_list = {}
    for label in tqdm(label_amount,"处理类别中："):
        # 确定每个类别拷贝的数量
        origin_amount = label_amount[label]
        # print(f"o:{origin_amount}")
        if origin_amount <= min_label_amount + 20:
            need_amount = origin_amount
        else:
            need_amount = min_label_amount + 20
        # print(f"n: {need_amount}")
        # 打乱列表,输出
        label_path = os.path.join(root_path, label)
        imgs_list = os.listdir(label_path)
        # print(imgs_list[:10])
        random.shuffle(imgs_list)
        output_list[label] = imgs_list[:need_amount]
        # print(output_list[label][:10])

    return output_list

if __name__ == "__main__":
    root_dir = r""
    output_dir = r""
    os.makedirs(output_dir, exist_ok=True)

    copy_dirs(root_dir, output_dir)

    label_copy_dict = balance_amount(root_dir)

    for label in label_copy_dict:
        old_path = os.path.join(root_dir, label)
        new_path = os.path.join(output_dir, label)
        for img_name in label_copy_dict[label]:
            old_img_path = os.path.join(old_path, img_name)
            new_img_path = os.path.join(new_path, img_name)
            shutil.copy(old_img_path, new_img_path)