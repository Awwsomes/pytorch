import os
import json
import shutil
from typing import Dict, Union
from tqdm import tqdm

def merge_category_dataset(
        category_mapping: Union[Dict, str],
        old_dataset_path: str,
        new_dataset_path: str
) -> None:
    """
    合并分类数据集类别，生成新的分类数据集

    Args:
        category_mapping: 类别映射关系，可以是字典 或 JSON文件路径
                          格式示例: {"新类别1": ["旧类别1", "旧类别2"], "新类别2": ["旧类别3"]}
        old_dataset_path: 原始分类数据集根路径（每个子文件夹为一个类别）
        new_dataset_path: 新合并后数据集的保存路径
    """
    # 1. 解析类别映射
    if isinstance(category_mapping, str):
        with open(category_mapping, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    else:
        mapping = category_mapping

    # 构建 旧类别 -> 新类别 的反向映射
    old2new = {}
    for new_cat, old_cats in mapping.items():
        for old_cat in old_cats:
            old2new[old_cat] = new_cat

    # 2. 创建新数据集根目录
    os.makedirs(new_dataset_path, exist_ok=True)

    # 3. 遍历原始数据集所有类别文件夹
    for old_cat in tqdm(os.listdir(old_dataset_path), "处理类别:"):
        old_cat_dir = os.path.join(old_dataset_path, old_cat)

        # 只处理文件夹
        if not os.path.isdir(old_cat_dir):
            continue

        # 不在映射中的旧类别直接跳过
        if old_cat not in old2new:
            print(f"跳过未映射类别: {old_cat}")
            continue

        # 对应新类别
        new_cat = old2new[old_cat]
        new_cat_dir = os.path.join(new_dataset_path, new_cat)
        os.makedirs(new_cat_dir, exist_ok=True)

        # 4. 复制该旧类别下所有文件到新类别文件夹
        for filename in os.listdir(old_cat_dir):
            old_file = os.path.join(old_cat_dir, filename)
            new_file = os.path.join(new_cat_dir, filename)

            if os.path.isfile(old_file):
                shutil.copy2(old_file, new_file)  # 保留元信息

    print(f"类别合并完成！新数据集已保存至: {new_dataset_path}")

if __name__ == "__main__":
    class_dataset_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_blue_mix3"
    output_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_blue_mix4"
    mapping = {
        "r1_blue": ["1"],
        "r2_blue": ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                                  "11", "12", "13", "14", "15", "16"],
        "fake_blue": ["17", "18", "19", "20",
                          "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    }
    merge_category_dataset(mapping, class_dataset_path, output_path)