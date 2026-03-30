import os
import shutil
from glob import glob

def convert_train_dataset_to_class_dataset(origin_root, new_root, class_names):
    """
    根据标签txt第一行的数字索引，映射到类别名列表，将图片分类到对应类别名文件夹
    :param origin_root: 原数据集根目录（内部必须有images和labels子文件夹）
    :param new_root: 新分类数据集的输出根目录
    :param class_names: 类别名列表（核心映射），如['cat','dog','bird']，标签第一行0→cat，1→dog
    """
    # 1. 定义关键路径和基础配置，检查输入合法性
    origin_images = os.path.join(origin_root, "images")
    origin_labels = os.path.join(origin_root, "labels")
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # 兼容的图片后缀
    max_index = len(class_names) - 1  # 最大有效标签索引（类别数-1）

    # 输入合法性检查
    if not class_names:
        print(f"❌ 错误：类别名列表不能为空，请输入有效类别名！")
        return
    if not os.path.exists(origin_root):
        print(f"❌ 错误：原数据集根目录不存在 -> {origin_root}")
        return
    if not os.path.exists(origin_images):
        print(f"❌ 错误：原数据集images文件夹不存在 -> {origin_images}")
        return
    if not os.path.exists(origin_labels):
        print(f"❌ 错误：原数据集labels文件夹不存在 -> {origin_labels}")
        return

    # 2. 动态创建新数据集的类别文件夹（以类别名命名，exist_ok=True避免重复创建）
    for cls_name in class_names:
        cls_dir = os.path.join(new_root, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
    # 打印类别映射关系（索引:类别名），方便核对
    class_mapping = {i: name for i, name in enumerate(class_names)}
    print(f"✅ 新数据集文件夹已创建/存在 -> {new_root}")
    print(f"✅ 类别映射关系（标签第一行数字→文件夹名）：{class_mapping}")
    print(f"✅ 兼容图片格式：{image_suffixes}\n")

    # 3. 初始化统计变量（动态生成，键为类别名）
    total_processed = 0  # 总成功处理数
    cls_count = {name: 0 for name in class_names}  # 按类别名统计数量
    error_count = 0  # 错误文件数
    error_files = []  # 错误文件记录（方便后续排查）

    # 4. 遍历所有标签txt文件（核心逻辑）
    txt_files = glob(os.path.join(origin_labels, "*.txt"))
    if not txt_files:
        print(f"⚠️  警告：labels文件夹下无txt标签文件 -> {origin_labels}")
        return
    print(f"开始处理，共发现 {len(txt_files)} 个标签文件...\n")

    for txt_path in txt_files:
        # 获取标签文件名（无后缀），用于匹配同名图片
        txt_name = os.path.basename(txt_path).split(".")[0]
        try:
            # 读取txt第一行，提取标签索引（去除首尾空白符）
            with open(txt_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            # 标签索引转整数（非数字直接报错）
            label = first_line.split()[0]
            label_index = int(label)
            # 检查索引是否在有效范围（0 ~ 类别数-1）
            if label_index < 0 or label_index > max_index:
                raise ValueError(f"标签索引超出有效范围[0, {max_index}]")
            # 映射到实际类别名
            target_cls = class_names[label_index]

            # 查找同名图片（兼容所有指定后缀，按顺序匹配）
            image_path = None
            for suffix in image_suffixes:
                temp_path = os.path.join(origin_images, txt_name + suffix)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
            if not image_path:
                raise FileNotFoundError("未找到同名图片（兼容jpg/png/bmp/webp）")

            # 复制图片到新数据集对应类别名文件夹（保留原文件名，copy2保留文件元信息）
            new_image_path = os.path.join(new_root, target_cls, os.path.basename(image_path))
            shutil.copy2(image_path, new_image_path)

            # 更新统计
            total_processed += 1
            cls_count[target_cls] += 1
            print(f"✅ 处理成功：{txt_name} -> 类别[{label_index}] {target_cls}")

        except ValueError as e:
            # 捕获：非数字标签、索引越界、空标签等情况
            error_info = f"{txt_name} | 标签错误：{str(e)}，原始标签值：{first_line}"
            error_files.append(error_info)
            error_count += 1
            print(f"❌ 处理失败：{error_info}")
        except FileNotFoundError as e:
            error_info = f"{txt_name} | {str(e)}"
            error_files.append(error_info)
            error_count += 1
            print(f"❌ 处理失败：{error_info}")
        except Exception as e:
            # 捕获其他未知错误（文件损坏、权限不足、中文路径等）
            error_info = f"{txt_name} | 未知错误：{str(e)[:50]}"  # 截断过长错误信息
            error_files.append(error_info)
            error_count += 1
            print(f"❌ 处理失败：{error_info}")

    # 5. 打印最终统计结果（直观显示类别名数量）
    print("\n" + "-" * 60)
    print("📊 分类完成 - 统计结果")
    print(f"总标签文件数：{len(txt_files)} | 成功处理数：{total_processed} | 失败文件数：{error_count}")
    print(f"各类别数量：{', '.join([f'{k}->{v}张' for k, v in cls_count.items()])}")
    if error_files:
        print(f"失败详情（前10条）：{error_files[:10]}")  # 只显示前10条，避免刷屏
        if len(error_files) > 10:
            print(f"⚠️  失败记录共{len(error_files)}条，以上仅显示前10条")
    print(f"📁 新分类数据集路径：{new_root}")
    print("-" * 60)

def classify_dataset_by_label(origin_root, new_root, class_names):
    """
    根据标签txt第一行的数字索引，映射到类别名列表，将图片分类到对应类别名文件夹
    :param origin_root: 原数据集根目录（内部必须有images和labels子文件夹）
    :param new_root: 新分类数据集的输出根目录
    :param class_names: 类别名列表
    """
    # 1. 定义关键路径和基础配置，检查输入合法性
    origin_images = os.path.join(origin_root, "images")
    origin_labels = os.path.join(origin_root, "labels")
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # 兼容的图片后缀
    max_index = len(class_names) - 1  # 最大有效标签索引（类别数-1）

    # 输入合法性检查
    if not class_names:
        print(f"❌ 错误：类别名列表不能为空，请输入有效类别名！")
        return
    if not os.path.exists(origin_root):
        print(f"❌ 错误：原数据集根目录不存在 -> {origin_root}")
        return
    if not os.path.exists(origin_images):
        print(f"❌ 错误：原数据集images文件夹不存在 -> {origin_images}")
        return
    if not os.path.exists(origin_labels):
        print(f"❌ 错误：原数据集labels文件夹不存在 -> {origin_labels}")
        return

    # 2. 动态创建新数据集的类别文件夹（以类别名命名，exist_ok=True避免重复创建）
    for cls_name in class_names:
        cls_dir = os.path.join(new_root, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
    # 打印类别映射关系（索引:类别名），方便核对
    class_mapping = {i: name for i, name in enumerate(class_names)}
    print(f"✅ 新数据集文件夹已创建/存在 -> {new_root}")
    print(f"✅ 类别映射关系（标签第一行数字→文件夹名）：{class_mapping}")
    print(f"✅ 兼容图片格式：{image_suffixes}\n")

    # 3. 初始化统计变量（动态生成，键为类别名）
    total_processed = 0  # 总成功处理数
    cls_count = {name: 0 for name in class_names}  # 按类别名统计数量
    error_count = 0  # 错误文件数
    error_files = []  # 错误文件记录（方便后续排查）

    # 4. 遍历所有标签txt文件（核心逻辑）
    txt_files = glob(os.path.join(origin_labels, "*.txt"))
    if not txt_files:
        print(f"⚠️  警告：labels文件夹下无txt标签文件 -> {origin_labels}")
        return
    print(f"开始处理，共发现 {len(txt_files)} 个标签文件...\n")

    for txt_path in txt_files:
        # 获取标签文件名（无后缀），用于匹配同名图片
        txt_name = os.path.basename(txt_path).split(".")[0]
        try:
            # 读取txt第一行，提取标签索引（去除首尾空白符）
            with open(txt_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            # 读取标签
            label = first_line.split()[0]
            # 检查标签是否在标签列表内
            if not label in class_names:
                print(f"{label} 不在class_names内：{txt_name}")
                continue

            # # 检查索引是否在有效范围（0 ~ 类别数-1）
            # if label_index < 0 or label_index > max_index:
            #     raise ValueError(f"标签索引超出有效范围[0, {max_index}]")
            # # 映射到实际类别名
            # target_cls = class_names[label_index]

            # 查找同名图片（兼容所有指定后缀，按顺序匹配）
            image_path = None
            for suffix in image_suffixes:
                temp_path = os.path.join(origin_images, txt_name + suffix)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
            if not image_path:
                raise FileNotFoundError("未找到同名图片（兼容jpg/png/bmp/webp）")

            # 复制图片到新数据集对应类别名文件夹（保留原文件名，copy2保留文件元信息）
            new_image_path = os.path.join(new_root, label, os.path.basename(image_path))
            shutil.copy2(image_path, new_image_path)

            # 更新统计
            total_processed += 1
            cls_count[label] += 1
            print(f"✅ 处理成功：{txt_name} -> 类别：{label}")

        except ValueError as e:
            # 捕获：非数字标签、索引越界、空标签等情况
            error_info = f"{txt_name} | 标签错误：{str(e)}，原始标签值：{first_line}"
            error_files.append(error_info)
            error_count += 1
            print(f"❌ 处理失败：{error_info}")
        except FileNotFoundError as e:
            error_info = f"{txt_name} | {str(e)}"
            error_files.append(error_info)
            error_count += 1
            print(f"❌ 处理失败：{error_info}")
        except Exception as e:
            # 捕获其他未知错误（文件损坏、权限不足、中文路径等）
            error_info = f"{txt_name} | 未知错误：{str(e)[:50]}"  # 截断过长错误信息
            error_files.append(error_info)
            error_count += 1
            print(f"❌ 处理失败：{error_info}")

    # 5. 打印最终统计结果（直观显示类别名数量）
    print("\n" + "-" * 60)
    print("📊 分类完成 - 统计结果")
    print(f"总标签文件数：{len(txt_files)} | 成功处理数：{total_processed} | 失败文件数：{error_count}")
    print(f"各类别数量：{', '.join([f'{k}->{v}张' for k, v in cls_count.items()])}")
    if error_files:
        print(f"失败详情（前10条）：{error_files[:10]}")  # 只显示前10条，避免刷屏
        if len(error_files) > 10:
            print(f"⚠️  失败记录共{len(error_files)}条，以上仅显示前10条")
    print(f"📁 新分类数据集路径：{new_root}")
    print("-" * 60)

if __name__ == "__main__":
    # -------------------------- 请在这里修改你的配置 --------------------------
    # 1. 原数据集根目录：内部必须有「images」和「labels」两个同级子文件夹
    ORIGIN_DATASET_ROOT = r"D:\A_myData\dataset\test_map50_cla_"  # Windows路径用r前缀，Linux/Mac用"/home/user/xxx"
    # 2. 新分类数据集输出目录：脚本自动创建，无需手动建
    NEW_DATASET_ROOT = r"D:\A_myData\dataset\test_map50_cla_2"
    # 3. 核心：类别名列表（标签第一行数字为索引，对应此列表）
    # 示例1：3类别 ['猫', '狗', '其他'] → 标签0→猫，1→狗，2→其他
    # 示例2：4类别 ['car', 'bike', 'person', 'bus'] → 标签0→car，1→bike，2→person，3→bus
    # 示例3：原固定类别 ['0', '1', '2'] → 还原成最初的数字文件夹
    CLASS_NAMES = ["0","1","2"]
    # -------------------------------------------------------------------------

    # 执行分类
    classify_dataset_by_label(ORIGIN_DATASET_ROOT, NEW_DATASET_ROOT, CLASS_NAMES)