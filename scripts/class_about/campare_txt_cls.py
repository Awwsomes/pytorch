import os
from tqdm import tqdm

path_txts_accurate = r"D:\A_myData\dataset\juanZhou_gazebo4-cls\val\labels"  # 人打的准确的标签
path_txts_predict = r"D:\A_myData\Pytorch\yolo11_cls\runs\classify\predict6\labels"   # 模型预测的标签

# 读取两个文件夹下的标签文件
list_txt_accurate = os.listdir(path_txts_accurate)
list_txt_predict = os.listdir(path_txts_predict)

amount_txt_file_accurate = len(list_txt_accurate)
amount_txt_file_predict = len(list_txt_predict)

amount_txt_accurate = 0
amount_predict_true = 0
amount_no_detection = 0
amount_detect_wrong = 0

# 统计手打的标签文件中标签总数
for txt_accurate in tqdm(list_txt_accurate):
    path_txt_accurate = os.path.join(path_txts_accurate, txt_accurate)
    with open(path_txt_accurate, 'r') as file_accurate:
        # 读取所有行到列表（每行是一个元素）
        lines_accurate = file_accurate.readlines()
        # 列表支持 len()，得到行数（即目标数量）
        amount_txt_accurate += len(lines_accurate)

# 遍历打的标签文件
for txt_accurate in tqdm(list_txt_accurate):
    # 判断没识别到（没预测标签文件）
    if not txt_accurate in list_txt_predict:
        print("\nNote:{} no detection.".format(txt_accurate))
        amount_no_detection += 1
        continue

    # 同时打开两个标签文件
    path_txt_accurate = os.path.join(path_txts_accurate,txt_accurate)
    path_txt_predict = os.path.join(path_txts_predict,txt_accurate)
    with open(path_txt_accurate,'r') as file_accurate , \
         open(path_txt_predict,'r') as file_predict:
        # 遍历打的标签文件的行
        for line_accurate in file_accurate:
            # 找预测的标签文件的行有没有这个结果
            label_accurate = line_accurate.split(" ")[0]
            have_label = [True for x in file_predict if str(int(x.split(" ")[1]) - 1) == label_accurate]
            if have_label:
                amount_predict_true += 1
            else:
                print("Note:{} predict wrong.".format(txt_accurate))
                amount_detect_wrong += 1

accuracy = amount_predict_true / (amount_predict_true+amount_detect_wrong)
print("--------------------------------")
print("标签总数:{}".format(amount_txt_file_accurate))
print("未检测到的标签数:{}".format(amount_no_detection))
print("检测错的标签数:{}".format(amount_detect_wrong))
print("检测正确的标签数：{}".format(amount_predict_true))
print(f"漏检率：{amount_no_detection/amount_txt_file_accurate}")
print("精确率：{:.3f}".format(accuracy))
print("正确率：{:.3f}".format(amount_predict_true/amount_txt_file_accurate))