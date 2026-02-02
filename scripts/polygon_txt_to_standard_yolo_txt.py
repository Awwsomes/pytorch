import os

def process_file(input_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件名（不含路径）
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            parts = line.strip().split()
            # if len(parts) != 9:
            #     continue  # 跳过格式不对的行

            label = parts[0]
            coords = list(map(float, parts[1:]))

            # blank = np.zeros((100,100),dtype=np.uint8)
            # cv2.circle(blank,(int(100*coords[0]),int(100*coords[1])),1,255,thickness=-1)
            # cv2.circle(blank, (100 * coords[2], 100 * coords[3]), 1, 255, thickness=-1)
            # cv2.circle(blank, (100 * coords[4], 100 * coords[5]), 1, 255, thickness=-1)
            # cv2.circle(blank, (100 * coords[6], 100 * coords[7]), 1, 255, thickness=-1)

            # 提取四个点 (x,y)
            points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            width = xmax - xmin
            height = ymax - ymin
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2

            # cv2.circle(blank, (100 * cx, 100 * cy), 1, 255, thickness=-1)
            # cv2.imshow("img",blank)
            # cv2.waitKey(0)

            # 写入结果：标签 序号 中心x 中心y 宽 高
            f_out.write(f"{label} {cx} {cy} {width} {height}\n")


if __name__ == "__main__":
    # 输入文件路径
    input_file = r"D:\A_myData\dataset\corner1\labels"  # 这里替换成你的源文件路径
    output_folder = r"D:\A_myData\dataset\corner1\labels_standard"  # 输出文件夹

    list_txt = os.listdir(input_file)
    for txt in list_txt:
        path_txt = os.path.join(input_file,txt)
        process_file(path_txt,output_folder)

    print("处理完成！")
