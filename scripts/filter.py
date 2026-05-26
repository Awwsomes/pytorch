import os
import shutil


# 定义路径（根据实际路径修改）
images_dir = r"D:\A_myData\RC26-Vision\dataset\corner11\images"  # 原始图片文件夹
json_dir = r"D:\A_myData\RC26-Vision\dataset\corner11\jsons"  # JSON文件夹
output_dir = r"D:\A_myData\RC26-Vision\dataset\corner11\images_fix"  # 目标输出文件夹

# 支持的图片扩展名（可根据需求添加，如.png、.jpeg等）
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

# 看看源文件夹存不存在
if not os.path.exists(images_dir):
    print("WARNING: {} not exist!".format(images_dir))
    exit(-1)
elif not os.path.exists(json_dir):
    print("WARNING: {} not exist!".format(json_dir))
    exit(-1)

# 创建目标文件夹（若不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历原始图片文件夹中的所有文件
for filename in os.listdir(images_dir):
    # 检查是否为支持的图片格式
    if filename.lower().endswith(image_extensions):
        # 提取文件名（不带扩展名）
        base_name = os.path.splitext(filename)[0]
        # 构造对应的JSON文件路径
        json_path = os.path.join(json_dir, f"{base_name}.json")

        # 检查JSON文件是否存在
        if os.path.isfile(json_path):
            # 构造源图片路径和目标路径
            src_image = os.path.join(images_dir, filename)
            dest_image = os.path.join(output_dir, filename)

            # 复制文件（保留元数据）
            shutil.copy2(src_image, dest_image)
            print(f"已复制: {filename} -> {output_dir}")