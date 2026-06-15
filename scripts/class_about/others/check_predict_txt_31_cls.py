import os
from os import mkdir

# 计算单个类别的精确率和召回率
def cal_accuracy_and_recall(path_root,label_predict):
    list_txt = os.listdir(path_root)
    # list_txt = ["juanZhou1_0.txt"]

    amount_predict_true = 0
    amount_predict_wrong = 0

    for txt in list_txt:
        path_txt = os.path.join(path_root,txt)
        with open(path_txt,'r') as txt_file:
            max_conv = 0.0
            label_max_conv = 0
            # 遍历每一行
            for idx,line in enumerate(txt_file):
                # 找置信度最高的一行
                conv = float(line.split(' ')[0])
                if conv > max_conv:
                    max_conv = conv
                    label_max_conv = int(line.split(' ')[1])
            # 检查预测标签和实际标签是否一致
            if label_max_conv-1 == label_predict:
                amount_predict_true += 1
            else:
                amount_predict_wrong += 1
                print("{}/{} Note:{} 识别错误为 {}".format(label_predict+1,31,txt,label_max_conv))

    return amount_predict_true,amount_predict_wrong

# 结果写入txt
def write_txt(path_output_txt,label_predict,accuracy,recall):
    # 打开txt文件
    with open(path_output_txt,'a+') as output_txt:   # a+ 模式，文末追加内容模式，+指同时还能读取，指的是光标移到了文件最后，无论读取还是写入都是在光标之后进行的
        # 光标移到开头才能读到全部内容
        output_txt.seek(0)
        lines = output_txt.readlines()
        num_of_lines = len(lines)
        if num_of_lines == 0:
            output_txt.write("Label Accuracy Recall\n")
        # 格式：类别序号 召回率 精确率
        output_txt.write("{} {} {}\n".format(label_predict,accuracy,recall))

# 计算31个类别
def cal_31_and_write(path_root,path_output_txt,path_images):
    all_predict_true = 0
    all_label_amount = 0

    for idx in range(31):
        # if idx == 0:
        #     part = os.path.join("exp", "labels")
        #     path_txt = os.path.join(path_root, part)
        # else:
        #     part = os.path.join("exp" + str(idx+1),"labels")
        #     path_txt = os.path.join(path_root,part)
        part = os.path.join("predict" + str(idx+3),"labels")
        path_txt = os.path.join(path_root,part)

        # 读取预测照片张数
        path_image = os.path.join(path_images,str(idx+1))
        list_img = os.listdir(path_image)
        list_img = [x for x in list_img if os.path.isfile(os.path.join(path_image,x))]
        amount_img = len(list_img)

        amount_predict_true,amount_predict_wrong = cal_accuracy_and_recall(path_txt,idx)

        recall = amount_predict_true / amount_img
        accuracy = amount_predict_true / (amount_predict_true + amount_predict_wrong)

        print("--------------------------")
        print("类别：{}".format(idx+1))
        print("图片总数：{}".format(amount_img))
        print("预测到标签总数：{}".format(amount_predict_true + amount_predict_wrong))
        print("识别正确数：{}".format(amount_predict_true))
        print("识别错误数：{}".format(amount_predict_wrong))
        print("召回率：{:.3f}".format(recall))  # ?
        print("精确率：{:.3f}".format(accuracy))
        print("--------------------------")

        all_predict_true += amount_predict_true
        all_label_amount += amount_img

        write_txt(path_output_txt, idx, accuracy, recall)

    # 打印总的正确率，错误个数
    print(f"总的标签个数：{all_label_amount}")
    print(f"总的识别对的标签个数：{all_predict_true}")
    print(f"总的识别错的标签个数：{all_label_amount - all_predict_true}")
    print(f"总的正确率：{all_predict_true / all_label_amount}")
    print(f"总的错误率：{(all_label_amount - all_predict_true) / all_label_amount}")

if __name__ == "__main__":
    path_root = r"D:\A_myData\Pytorch\yolo11_cls\runs\classify\predict3"  # 模型预测出来的标签文件夹
    path_output_txt = r"D:\A_myData\Pytorch\src\data\yolo11-cls_6000.txt"  # 存储输出数据的txt文件
    path_images = r"D:\A_myData\dataset\juanZhou_gazebo6-cls"   # 存放被预测图片的文件夹

    path_output_dir,_ = os.path.split(path_output_txt)
    if not os.path.exists(path_output_dir):
        os.mkdir(path_output_dir)

    cal_31_and_write(path_root,path_output_txt,path_images)