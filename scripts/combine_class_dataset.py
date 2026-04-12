import os
import random
import shutil

def count_images(dataset_dir, dataset_name="数据集"):
    """
    统计数据集中各类别的图片数量，并格式化打印
    返回: 字典 {类别名: [图片路径列表]}, 总数量
    """
    class_images = {}
    total_count = 0

    if not os.path.exists(dataset_dir):
        print(f"警告: 路径 {dataset_dir} 不存在")
        return class_images, 0

    # 遍历类别文件夹
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # 获取该类别下所有图片
        images = []
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append(os.path.join(class_path, img_file))

        class_images[class_name] = images
        total_count += len(images)

    # ------------------ 格式化打印统计信息 ------------------
    print(f"📂 {dataset_name} 统计:")
    if not class_images:
        print("  (空)")
    else:
        col_count = 0
        sorted_classes = sorted(class_images.keys())
        for cls in sorted_classes:
            cnt = len(class_images[cls])
            print_str = f"[{cls}] {cnt}张"

            if col_count < 2:
                print(f"  {print_str:<18}", end="")
                col_count += 1
            else:
                print(f"  {print_str}")
                col_count = 0
        if col_count != 0:
            print("")

    print(f"  📊 总计: {total_count} 张\n")
    return class_images, total_count


def print_combined_stats(data1, data2, total1, total2):
    """打印两个数据集的联合统计概览"""
    all_classes = set(data1.keys()).union(set(data2.keys()))
    print("=" * 60)
    print("📈 联合数据概览 (格式: [类别名] D1+D2=总量)")
    print("-" * 60)

    sorted_classes = sorted(all_classes)
    col_count = 0

    for cls in sorted_classes:
        cnt1 = len(data1.get(cls, []))
        cnt2 = len(data2.get(cls, []))
        sum_cnt = cnt1 + cnt2

        print_str = f"[{cls}] {cnt1}+{cnt2}={sum_cnt}"

        if col_count < 2:
            print(f"{print_str:<28}", end="")
            col_count += 1
        else:
            print(f"{print_str}")
            col_count = 0

    if col_count != 0:
        print("")

    print("-" * 60)
    print(f"数据集1总计: {total1} | 数据集2总计: {total2} | 合计: {total1 + total2}")
    print("=" * 60 + "\n")


def merge_and_copy(keep_data, keep_name, supp_data, supp_name, target_counts, output_dir):
    """执行具体的合并和拷贝逻辑"""
    print("\n" + "=" * 30)
    print("🚀 开始合并图片...")

    os.makedirs(output_dir, exist_ok=True)
    all_classes = sorted(target_counts.keys())

    for cls in all_classes:
        target = target_counts[cls]
        keep_imgs = keep_data.get(cls, [])
        supp_imgs = supp_data.get(cls, [])

        out_class_dir = os.path.join(output_dir, cls)
        os.makedirs(out_class_dir, exist_ok=True)

        # 复制保留数据集
        print(f"\n处理 [{cls}]: 保留 {len(keep_imgs)} 张", end="")
        for img_path in keep_imgs:
            filename = f"{keep_name}_{os.path.basename(img_path)}"
            shutil.copy2(img_path, os.path.join(out_class_dir, filename))

        # 随机补充图片
        needed = target - len(keep_imgs)
        if needed > 0:
            print(f" | 需补充 {needed} 张", end="")
            if len(supp_imgs) < needed:
                print(f" (补充源仅有{len(supp_imgs)}张，全取)", end="")
                selected = supp_imgs
            else:
                selected = random.sample(supp_imgs, needed)

            for img_path in selected:
                filename = f"{supp_name}_{os.path.basename(img_path)}"
                shutil.copy2(img_path, os.path.join(out_class_dir, filename))

        final_count = len(os.listdir(out_class_dir))
        print(f" | 完成: {final_count} 张")

    print("\n" + "=" * 30)
    print(f"✅ 全部完成！结果保存至: {output_dir}")


def merge_classification_datasets(d1_path, d2_path, out_path, combine_label_list):
    """主逻辑流程"""
    # 1. 统计
    data1, total1 = count_images(d1_path, "数据集 1")
    data2, total2 = count_images(d2_path, "数据集 2")

    all_classes = set(data1.keys()).union(set(data2.keys()))
    if not all_classes:
        print("❌ 错误: 未在两个数据集中找到任何图片")
        return

    # 2. 打印联合概览
    print_combined_stats(data1, data2, total1, total2)

    # ====================== 新增：指定合并类别列表 ======================
    print(f"检测到的所有类别: {', '.join(sorted(all_classes))}")
    while True:
        valid_classes = []
        invalid_classes = []
        for c in combine_label_list:
            if c in all_classes:
                valid_classes.append(c)
            else:
                invalid_classes.append(c)

        if invalid_classes:
            print(f"  警告: 以下类别未找到 -> {', '.join(invalid_classes)}")

        if valid_classes:
            target_classes_list = valid_classes
            print(f"  即将合并类别: {', '.join(target_classes_list)}")
            break
        else:
            print("  错误: 未输入任何有效类别，请重新输入")
    # ==================================================================

    # 3. 选择保留数据集
    while True:
        choice = input("请选择完全保留哪一个数据集 (输入 1 或 2): ")
        if choice == '1':
            keep_data, keep_name = data1, "D1"
            supp_data, supp_name = data2, "D2"
            break
        elif choice == '2':
            keep_data, keep_name = data2, "D2"
            supp_data, supp_name = data1, "D1"
            break
        print("输入无效，请重新输入")

    # 4. 输入统一目标数量 (注意：这里将 all_classes 替换为 target_classes_list)
    # max_keep_count = max(len(keep_data.get(cls, [])) for cls in all_classes) # 旧代码
    max_keep_count = max(len(keep_data.get(cls, [])) for cls in target_classes_list)  # 新代码

    while True:
        try:
            target = int(input(f"\n请输入所有类别的最终目标数量 (必须 ≥ {max_keep_count}): "))
            # if all(len(keep_data.get(cls, [])) <= target for cls in all_classes): # 旧代码
            if all(len(keep_data.get(cls, [])) <= target for cls in target_classes_list):  # 新代码
                # target_counts = {cls: target for cls in all_classes} # 旧代码
                target_counts = {cls: target for cls in target_classes_list}  # 新代码
                break
            print("  错误: 目标数量小于保留数据集中的某些类别数量")
        except ValueError:
            print("  错误: 请输入整数")

    # 5. 执行合并
    merge_and_copy(keep_data, keep_name, supp_data, supp_name, target_counts, out_path)

if __name__ == "__main__":

    # ====================== 在这里直接修改你的路径 ======================
    dataset1_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_gazebo_real"
    dataset2_path = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_4_9\juanZhou_car4"
    output_path = r"D:\A_myData\RC26-Vision\dataset\test1"
    combine_label_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                          "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                          "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
    # ====================================================================

    merge_classification_datasets(dataset1_path, dataset2_path, output_path, combine_label_list)