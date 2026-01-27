import os
import cv2


def parse_txt_content(content_list):
    """从文本内容列表中解析类别和坐标"""
    crops = []
    i = 0
    while i < len(content_list):
        # 检查是否有足够数据
        if i + 9 > len(content_list):
            print(f"[Warning] Incomplete data at position {i}, skipping...")
            break

        # 解析类别ID（支持0及正整数）
        cls_str = content_list[i].strip()
        if not cls_str.isdigit():
            print(f"[Warning] Invalid class ID at position {i}, skipping...")
            i += 1
            continue
        cls_id = int(cls_str)

        # 解析坐标（必须包含8个数值）
        coords = content_list[i + 1:i + 9]
        if any(not c.replace('.', '', 1).isdigit() for c in coords):
            print(f"[Warning] Invalid coordinates at position {i + 1}, skipping...")
            i += 1
            continue
        coords = list(map(float, coords))

        # 验证坐标数量
        if len(coords) != 8:
            print(f"[Warning] Expected 8 coordinates at position {i + 1}, found {len(coords)}, skipping...")
            i += 1
            continue

        crops.append((cls_id, coords))
        i += 9  # 跳过已处理数据

    return crops


def calculate_bounding_box(coords):
    """根据四个顶点坐标计算最小包围框"""
    x_coords = coords[::2]  # 每隔一个元素取x坐标
    y_coords = coords[1::2]  # 每隔一个元素取y坐标
    xmin = int(min(x_coords))
    xmax = int(max(x_coords))
    ymin = int(min(y_coords))
    ymax = int(max(y_coords))
    return (xmin, ymin, xmax, ymax)


def check_the_dataset(image_dir, txt_dir, output_dir):
    """
    裁剪出框内的图像，根据类别存放在对应的类别文件夹下
    可用于检查标
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_filename in os.listdir(image_dir):
        if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_filename)
        base_name = os.path.splitext(img_filename)[0]
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(txt_dir, txt_filename)

        if not os.path.isfile(txt_path):
            print(f"[Warning] No txt file found for {img_filename}, skipping...")
            continue

        # 读取图像并获取真实尺寸
        image = cv2.imread(img_path)
        if image is None:
            print(f"[Error] Failed to read {img_filename}, skipping...")
            continue

        height, width = image.shape[:2]  # 动态获取图像尺寸

        # 读取并解析txt文件
        with open(txt_path, 'r') as f:
            content = f.read().replace('\t', ' ').replace('\n', ' ').split()
        content = [c.strip() for c in content if c.strip()]

        crops = parse_txt_content(content)

        for idx,crop in enumerate(crops):
            cls_id, coords = crop
            # 将归一化坐标转换为像素坐标
            pixel_coords = []
            for i in range(8):
                if i % 2 == 0:  # 偶数位为x坐标
                    pixel_coords.append(coords[i] * width)
                else:  # 奇数位为y坐标
                    pixel_coords.append(coords[i] * height)

            # 计算最小包围框
            x1, y1, x2, y2 = calculate_bounding_box(pixel_coords)

            # 边界检查（防止负值）
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            # 创建类别目录（支持class_0）
            class_dir = os.path.join(output_dir, f"class_{cls_id}")
            os.makedirs(class_dir, exist_ok=True)

            # 生成输出文件名
            output_filename = f"{base_name}_{cls_id}_{idx}.jpg"
            # output_filename = img_filename
            output_path = os.path.join(class_dir, output_filename)

            # 裁剪并保存图像
            # print(f"Processing: {x1},{y1} - {x2},{y2}")
            cropped_img = image[y1:y2, x1:x2]  # 修正切片语法
            cv2.imwrite(output_path, cropped_img)
            # print(f"[Success] Saved {output_filename} to {class_dir}")

# 检查数据集, 输出一个文件夹,内有每个类别的子文件夹,每个子文件夹中有同类别的照片, 照片没有resize
if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    image_dir = r"D:\A_myData\dataset\juanZhou_debug\images"
    txt_dir = r"D:\A_myData\dataset\juanZhou_debug\labels"
    output_dir = r"D:\A_myData\dataset\juanZhou_debug\check"

    # 执行裁剪
    check_the_dataset(image_dir, txt_dir, output_dir)