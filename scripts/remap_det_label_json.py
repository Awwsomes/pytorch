import json
import os
from pathlib import Path


def remap_labelme_labels(input_dir: str, output_dir: str, label_mapping: dict) -> dict:
    """
    批量重映射 labelme JSON 标注文件中的类别标签

    Args:
        input_dir: 输入 labelme JSON 文件所在文件夹路径
        output_dir: 输出重映射后 JSON 文件的文件夹路径
        label_mapping: 类别映射关系字典，格式为 {"新标签名": ["旧标签1", "旧标签2", ...]}
                       支持多个旧标签合并映射为同一个新标签

    Returns:
        dict: 统计信息，包含处理文件数、替换标签总数、未匹配标签统计等
    """
    # 构建反向映射表（旧标签 → 新标签），提升查找效率
    reverse_mapping = {}
    for new_label, old_labels in label_mapping.items():
        # 校验值必须为列表格式
        if not isinstance(old_labels, list):
            raise TypeError(f"映射表错误：新标签「{new_label}」对应的值必须是旧标签组成的列表")
        for old_label in old_labels:
            # 检测重复映射冲突，避免逻辑混乱
            if old_label in reverse_mapping:
                raise ValueError(
                    f"映射冲突：旧标签「{old_label}」同时被映射到 "
                    f"「{reverse_mapping[old_label]}」和「{new_label}」，请检查映射表"
                )
            reverse_mapping[old_label] = new_label

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 统计变量
    stats = {
        "total_files": 0,
        "total_shapes": 0,
        "remapped_shapes": 0,
        "unmatched_labels": {},
        "processed_files": []
    }

    # 遍历输入目录下所有 json 文件
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    for json_file in json_files:
        # 读取原始 JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "shapes" not in data:
            continue

        stats["total_files"] += 1

        # 遍历所有标注形状，替换标签
        for shape in data["shapes"]:
            old_label = shape["label"]
            stats["total_shapes"] += 1

            if old_label in reverse_mapping:
                shape["label"] = reverse_mapping[old_label]
                stats["remapped_shapes"] += 1
            else:
                # 记录未匹配的标签
                print(f"[Info] 未在映射表中匹配到的标签:{json_file}")
                stats["unmatched_labels"][old_label] = stats["unmatched_labels"].get(old_label, 0) + 1

        # 保存到输出目录
        output_path = Path(output_dir) / json_file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        stats["processed_files"].append(json_file.name)

    # 打印统计结果
    print("=" * 50)
    print(f"处理完成，共处理文件: {stats['total_files']} 个")
    print(f"总标注框数量: {stats['total_shapes']} 个")
    print(f"成功映射标签数量: {stats['remapped_shapes']} 个")

    if stats["unmatched_labels"]:
        print("\n未在映射表中匹配到的标签:")
        for label, count in stats["unmatched_labels"].items():
            print(f"  - {label}: {count} 个")
    else:
        print("\n所有标签均成功匹配映射表")
    print("=" * 50)

    return stats


if __name__ == "__main__":
    # ========== 配置区 ==========
    INPUT_JSON_DIR = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det_blue2\jsons_3"    # 输入文件夹
    OUTPUT_JSON_DIR = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det_blue2\jsons_3_fix"  # 输出文件夹
    label_map = {
        "r1_blue": ["1"],
        "r2_blue": ["2"],
        "fake_blue": ["3"]
    }
    # ============================

    # 执行映射
    remap_labelme_labels(INPUT_JSON_DIR, OUTPUT_JSON_DIR, label_map)