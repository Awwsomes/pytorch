import os

def delete_unmatched_txt(folder_path, txts_path, image_extensions=[".jpg", ".jpeg", ".png", ".bmp", ".gif"], test_mode=True):
    """
    删除无匹配名字图像的txt标签
    :param folder_path: 存放图像的文件夹
    :param txts_path: 存放txt的文件夹
    :param image_extensions: 受支持的图像扩展名列表 [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    :param test_mode: True则仅统计不删除，False会删除
    :return: none
    """
    # 步骤1：收集所有图片的文件名（不含后缀）
    image_names = set()  # 用集合存储，查询更快

    for filename in os.listdir(folder_path):
        # 获取文件后缀（转小写，避免大小写问题，如.JPG和.jpg）
        file_ext = os.path.splitext(filename)[1].lower()

        # 判断是否是图片文件
        if file_ext in image_extensions:
            # 提取文件名（不含后缀）并加入集合
            image_name = os.path.splitext(filename)[0]
            image_names.add(image_name)

    print(f"在文件夹中找到 {len(image_names)} 个图片文件")

    # 步骤2：遍历所有txt文件，判断是否有对应图片
    deleted_count = 0  # 统计删除的txt数量
    for filename in os.listdir(txts_path):
        # 判断是否是txt文件
        if filename.endswith(".txt"):
            # 提取txt的文件名（不含后缀）
            txt_name = os.path.splitext(filename)[0]

            # 判断对应的图片是否存在
            if txt_name not in image_names:
                # 构建txt文件的完整路径
                txt_path = os.path.join(txts_path, filename)

                if test_mode:
                    print(f"【测试模式】将要删除：{filename}（无对应图片）")
                else:
                    # 实际删除文件
                    os.remove(txt_path)
                    print(f"已删除：{filename}（无对应图片）")
                deleted_count += 1

    # 输出统计结果
    if test_mode:
        print(f"\n测试完成！共找到 {deleted_count} 个需要删除的多余txt文件")
        print("确认无误后，将 test_mode 改为 False 即可执行实际删除")
    else:
        print(f"\n删除完成！共删除 {deleted_count} 个多余的txt文件")

if __name__ == "__main__":
    # -------------------------- 请根据你的情况修改以下配置 --------------------------
    # 1. 图片和txt文件所在的文件夹路径（默认是当前文件夹，可改为绝对路径，如"D:/data"）
    folder_path = r"D:\A_myData\dataset\juanZhou_gazebo6-cls"
    txts_path = r"D:\A_myData\Pytorch\yolo11_cls\runs\classify\predict\labels_true"

    # 2. 需要匹配的图片后缀（常见格式已包含，可根据需要添加/删除）
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    # 3. 是否开启"测试模式"（True=只显示要删除的txt，不实际删除；False=实际删除）
    test_mode = True  # 建议先设为True，确认无误后再改为False

    delete_unmatched_txt(folder_path,txts_path,image_extensions,test_mode)