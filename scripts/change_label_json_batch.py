import os
import json
from typing import List, Dict, Tuple

def replace_labelme_labels(
        input_dir: str,
        output_dir: str,
        old_labels: List[str],
        new_labels: List[str]
) -> Tuple[int, int, int]:
    """
    批量替换Labelme格式JSON文件中的类别标签

    Args:
        input_dir: 存放原始JSON文件的文件夹路径
        output_dir: 保存修改后JSON文件的文件夹路径
        old_labels: 需要被替换的旧类别标签列表
        new_labels: 替换后的新类别标签列表（与old_labels顺序一一对应）

    Returns:
        Tuple[int, int, int]: 处理的文件总数，修改的文件数，修改的标签总数

    Raises:
        ValueError: 当old_labels和new_labels长度不相等时抛出
        FileNotFoundError: 当输入文件夹不存在时抛出
    """
    # 输入验证
    if len(old_labels) != len(new_labels):
        raise ValueError("旧标签列表和新标签列表的长度必须相等")

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 创建标签映射字典，提高查找效率
    label_map: Dict[str, str] = dict(zip(old_labels, new_labels))

    # 统计变量
    total_files = 0
    modified_files = 0
    total_modified_labels = 0

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.json'):
            total_files += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # 读取JSON文件
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 检查是否有shapes字段
                if 'shapes' not in data:
                    print(f"警告: 文件 {filename} 中没有找到shapes字段，跳过")
                    continue

                # 遍历所有标注形状并替换标签
                file_modified = False
                modified_labels_count = 0

                for shape in data['shapes']:
                    if 'label' in shape and shape['label'] in label_map:
                        old_label = shape['label']
                        new_label = label_map[old_label]
                        shape['label'] = new_label
                        file_modified = True
                        modified_labels_count += 1

                # 如果文件被修改过，保存到输出文件夹
                if file_modified:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    modified_files += 1
                    total_modified_labels += modified_labels_count
                    print(f"已修改: {filename} (替换了 {modified_labels_count} 个标签)")
                else:
                    # 如果没有修改，也复制一份到输出文件夹（保持文件完整性）
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"未修改: {filename}")

            except json.JSONDecodeError:
                print(f"错误: 文件 {filename} 不是有效的JSON格式，跳过")
            except Exception as e:
                print(f"错误: 处理文件 {filename} 时发生异常: {str(e)}")

    # 输出统计结果
    print("\n" + "=" * 50)
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"修改的文件数: {modified_files}")
    print(f"总共修改的标签数: {total_modified_labels}")
    print(f"修改后的文件已保存到: {output_dir}")
    print("=" * 50)

    return total_files, modified_files, total_modified_labels


# 使用示例
if __name__ == "__main__":
    # 配置参数
    INPUT_FOLDER = "D:\A_myData\RC26-Vision\dataset\corner11\jsons"  # 原始JSON文件所在文件夹
    OUTPUT_FOLDER = "D:\A_myData\RC26-Vision\dataset\corner11\jsons_blue"  # 修改后JSON文件保存文件夹

    # 要替换的标签列表（顺序一一对应）
    OLD_LABELS = ["corner", "trash"]
    NEW_LABELS = ["trash", "corner"]

    # 调用函数
    try:
        total, modified, labels_changed = replace_labelme_labels(
            input_dir=INPUT_FOLDER,
            output_dir=OUTPUT_FOLDER,
            old_labels=OLD_LABELS,
            new_labels=NEW_LABELS
        )
    except Exception as e:
        print(f"程序执行失败: {str(e)}")