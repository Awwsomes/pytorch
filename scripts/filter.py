import os
import shutil
import json

# 定义路径（根据实际路径修改）
images_dir = r"D:\A_myData\RC26-Vision\dataset\wuQiTou5\images"  # 原始图片文件夹
json_dir = r"D:\A_myData\RC26-Vision\dataset\wuQiTou5\corner_jsons"  # 标签文件夹
output_dir = r"D:\A_myData\RC26-Vision\dataset\wuQiTou5\images_fixed"  # 目标输出文件夹

# 支持的图片扩展名（可根据需求添加）
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
# 支持的标签扩展名
label_exts = (".json", ".txt")

# 统计信息
total_images = 0
copied_images = 0
skipped_empty_labels = 0
skipped_no_labels = 0
skipped_invalid_json = 0  # 新增：统计格式错误的JSON文件

# 检查源文件夹是否存在
if not os.path.exists(images_dir):
    print(f"ERROR: 图片文件夹不存在: {images_dir}")
    exit(-1)
elif not os.path.exists(json_dir):
    print(f"ERROR: 标签文件夹不存在: {json_dir}")
    exit(-1)

# 创建目标文件夹（若不存在）
os.makedirs(output_dir, exist_ok=True)


def is_valid_label_file(label_path):
    """
    检查标签文件是否有效（非空且包含实际标注）
    返回: (is_valid, skip_reason)
    """
    file_ext = os.path.splitext(label_path)[1].lower()

    # 先快速检查文件大小
    if os.path.getsize(label_path) == 0:
        return False, "文件大小为0"

    try:
        # 处理TXT标签文件
        if file_ext == ".txt":
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return False, "仅含空白字符"
            return True, None

        # 处理JSON标签文件（LabelMe格式）
        elif file_ext == ".json":
            with open(label_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return False, "JSON格式错误"

            # 关键修复：检查shapes字段是否为空数组
            if "shapes" not in data:
                return False, "缺少shapes字段"
            if len(data["shapes"]) == 0:
                return False, "shapes数组为空（无标注）"

            return True, None

        # 不支持的扩展名
        else:
            return False, f"不支持的标签格式: {file_ext}"

    except Exception as e:
        return False, f"读取失败: {str(e)}"


# 遍历原始图片文件夹中的所有文件
for filename in os.listdir(images_dir):
    # 检查是否为支持的图片格式
    if filename.lower().endswith(image_extensions):
        total_images += 1
        # 提取文件名（不带扩展名）
        base_name = os.path.splitext(filename)[0]

        has_valid_label = False
        for label_ext in label_exts:
            # 构造对应的标签文件路径
            label_path = os.path.join(json_dir, f"{base_name}{label_ext}")

            # 检查标签文件是否存在
            if os.path.isfile(label_path):
                is_valid, skip_reason = is_valid_label_file(label_path)

                if is_valid:
                    has_valid_label = True
                    break
                else:
                    print(f"跳过无效标签: {os.path.basename(label_path)} - 原因: {skip_reason}")
                    if "JSON格式错误" in skip_reason:
                        skipped_invalid_json += 1
                    else:
                        skipped_empty_labels += 1
                    continue

        if has_valid_label:
            # 构造源图片路径和目标路径
            src_image = os.path.join(images_dir, filename)
            dest_image = os.path.join(output_dir, filename)

            # 复制文件（保留元数据）
            shutil.copy2(src_image, dest_image)
            copied_images += 1
            print(f"已复制: {filename}")
        else:
            skipped_no_labels += 1
            print(f"跳过无有效标签的图片: {filename}")

# 输出最终统计信息
print("\n" + "=" * 60)
print("处理完成！统计信息：")
print(f"总图片数: {total_images}")
print(f"成功复制: {copied_images}")
print(f"跳过（无标签）: {skipped_no_labels}")
print(f"跳过（空/无效标签）: {skipped_empty_labels}")
print(f"跳过（JSON格式错误）: {skipped_invalid_json}")
print("=" * 60)