import os

def count_label_cls(root_path:str) -> (dict, int, list, int, list, float, int):
    list_dir = os.listdir(root_path)
    list_dir = [x for x in list_dir if os.path.isdir(os.path.join(root_path, x))]
    try:
        list_dir.sort(key=int)
    except:
        list_dir.sort()
    # print(list_dir)
    # list_dir = [x for x in list_dir if x in labels]

    label_amount = {}
    min_amount, max_amount, sum_amount = 0, 0, 0
    min_amount_idx_list, max_amount_idx_list = [], []

    for i, label_dir in enumerate(list_dir):
        # if label_dir not in labels:
        #     print(f"[Info] {label_dir} 不在类别列表内，不统计")
        list_imgs = os.listdir(os.path.join(root_path, label_dir))
        list_imgs = [x for x in list_imgs if os.path.splitext(x)[1] in [".png", ".jpg", ".jpeg", ".bmp"]]
        imgs_amount = len(list_imgs)
        label_amount[label_dir] = imgs_amount
        if i == 0:
            min_amount = imgs_amount
            max_amount = imgs_amount
        else:
            if imgs_amount > max_amount:
                max_amount = imgs_amount
                max_amount_idx_list.clear()
                max_amount_idx_list.append(label_dir)
            elif imgs_amount < min_amount:
                min_amount = imgs_amount
                min_amount_idx_list.clear()
                min_amount_idx_list.append(label_dir)
            elif imgs_amount == max_amount:
                max_amount_idx_list.append(label_dir)
            elif imgs_amount == min_amount:
                min_amount_idx_list.append(label_dir)
        sum_amount += imgs_amount

    return label_amount, min_amount, min_amount_idx_list, max_amount, max_amount_idx_list, sum_amount / len(label_amount), sum_amount

if __name__ == "__main__":

    input_path = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_real1"
    label_amount,min_amount,min_amount_idx_list,max_amount,max_amount_idx_list,average_amount,sum_amount = count_label_cls(input_path)

    print("-----------统计结果-----------")
    print("---------各类别统计结果--------")
    for idx, key in enumerate(label_amount):
        print(f"类别{key}: {label_amount[key]} ", end="")
        if (idx+1) % 3 == 0 or idx == len(label_amount) - 1:
            print(end="\n")
    print("----------------------------")
    print(f"最小值: {min_amount} 对应类别：{min_amount_idx_list}")
    print(f"最大值: {max_amount} 对应类别：{max_amount_idx_list}")
    print(f"均值: {average_amount:.3f}")
    print(f"总数: {sum_amount}")