import os
import shutil
import random


def extract_yolo_dataset(
        dataset_root: str,
        class_ids: list[int],
        counts: list[int],
        class_names: list[str],
        output_root: str,
        random_seed: int = 42,
        copy_full_label: bool = True
) -> None:
    """
    从YOLO格式检测数据集中提取特定类别指定数量的图片和标签

    Args:
        dataset_root: 原数据集根目录（需包含`images`和`labels`子文件夹）
        class_ids: 需要提取的类别序号列表（如[0, 2]）
        counts: 对应类别所需数量列表（需与class_ids长度一致，如[100, 50]）
        class_names: 类别映射列表（索引对应类别序号，如['person', 'car', 'dog']）
        output_root: 输出根目录（将自动创建`images`和`labels`子文件夹）
        random_seed: 随机种子，用于打乱图片顺序确保抽取均匀
        copy_full_label: 是否拷贝完整标签文件（True=保留所有类别，False=仅保留目标类别）
    """
    # -------------------------- 1. 输入参数校验 --------------------------
    if len(class_ids) != len(counts):
        raise ValueError(f"类别序号列表长度({len(class_ids)})与数量列表长度({len(counts)})不一致")

    for cid in class_ids:
        if cid < 0 or cid >= len(class_names):
            raise ValueError(f"类别序号 {cid} 超出类别映射列表范围（列表长度: {len(class_names)}）")

    # -------------------------- 2. 目录初始化 --------------------------
    # 原数据集目录
    input_img_dir = os.path.join(dataset_root, "images")
    input_label_dir = os.path.join(dataset_root, "labels")
    if not os.path.exists(input_img_dir):
        raise FileNotFoundError(f"原数据集图片目录不存在: {input_img_dir}")
    if not os.path.exists(input_label_dir):
        raise FileNotFoundError(f"原数据集标签目录不存在: {input_label_dir}")

    # 输出数据集目录
    output_img_dir = os.path.join(output_root, "images")
    output_label_dir = os.path.join(output_root, "labels")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # -------------------------- 3. 图片收集与打乱 --------------------------
    valid_img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
    img_files = [f for f in os.listdir(input_img_dir) if f.endswith(valid_img_exts)]
    if not img_files:
        raise ValueError(f"原数据集图片目录中未找到有效图片: {input_img_dir}")

    # 打乱图片顺序，确保随机抽取
    random.seed(random_seed)
    random.shuffle(img_files)

    # -------------------------- 4. 核心提取逻辑 --------------------------
    count_dict = {cid: 0 for cid in class_ids}  # 类别计数器
    processed_imgs = set()  # 已处理图片记录（避免重复）

    print(f"开始提取 | 目标: {dict(zip([class_names[cid] for cid in class_ids], counts))}")

    for img_file in img_files:
        # 提前终止：所有类别已达到目标数量
        if all(count_dict[cid] >= counts[i] for i, cid in enumerate(class_ids)):
            print("\n所有类别已达到目标数量，提前结束提取")
            break

        # 匹配标签文件
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        input_label_path = os.path.join(input_label_dir, label_file)
        if not os.path.exists(input_label_path):
            continue  # 无对应标签，跳过

        # 读取并解析标签
        with open(input_label_path, "r", encoding="utf-8") as f:
            label_lines = [line.strip() for line in f if line.strip()]

        # 检查标签是否包含目标类别（且未达数量）
        target_classes_in_img = []
        filtered_lines = []  # 仅当copy_full_label=False时使用
        for line in label_lines:
            cid = int(line.split()[0])
            if cid in class_ids:
                idx = class_ids.index(cid)
                if count_dict[cid] < counts[idx]:
                    target_classes_in_img.append(cid)
                if not copy_full_label:
                    filtered_lines.append(line)  # 仅保留目标类别
            elif copy_full_label:
                filtered_lines.append(line)  # 保留所有类别

        # 若无目标类别，跳过
        if not target_classes_in_img:
            continue

        # -------------------------- 5. 拷贝文件 --------------------------
        # 拷贝图片
        shutil.copy2(
            os.path.join(input_img_dir, img_file),
            os.path.join(output_img_dir, img_file)
        )
        # 拷贝标签（根据参数决定是否过滤）
        output_label_path = os.path.join(output_label_dir, label_file)
        with open(output_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_lines) + "\n")

        # -------------------------- 6. 更新状态 --------------------------
        for cid in target_classes_in_img:
            idx = class_ids.index(cid)
            if count_dict[cid] < counts[idx]:
                count_dict[cid] += 1
        processed_imgs.add(img_file)

        # 打印进度
        if len(processed_imgs) % 10 == 0:
            print(f"已处理 {len(processed_imgs)} 张 | 当前计数: {count_dict}")

    # -------------------------- 7. 结果统计 --------------------------
    print("\n" + "=" * 50)
    print(f"提取完成！共处理 {len(processed_imgs)} 张图片")
    print("最终各类别统计:")
    for i, cid in enumerate(class_ids):
        print(f"  - {class_names[cid]} (ID:{cid}): {count_dict[cid]}/{counts[i]}")
    print(f"输出目录: {output_root}")
    print("=" * 50)


if __name__ == "__main__":
    # ====================== 配置区域（请根据实际情况修改） ======================
    DATASET_ROOT = "D:\A_myData\RC26-Vision\dataset\juanZhou_obb1"  # 原数据集根目录
    CLASS_IDS = [0]  # 需要提取的类别序号
    COUNTS = [300]  # 对应类别的提取数量
    CLASS_NAMES = ["1"]  # 类别映射（索引=类别ID）
    OUTPUT_ROOT = "D:\A_myData\RC26-Vision\dataset\juanZhou_obb5"  # 输出根目录
    # ===========================================================================

    # 执行提取
    extract_yolo_dataset(
        dataset_root=DATASET_ROOT,
        class_ids=CLASS_IDS,
        counts=COUNTS,
        class_names=CLASS_NAMES,
        output_root=OUTPUT_ROOT,
        random_seed=34587,
        copy_full_label=True  # 设为False可仅保留目标类别的标签
    )