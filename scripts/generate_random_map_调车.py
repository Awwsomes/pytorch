import os
import random
import json

def generate_random_map_list(block_list=None) -> list:

    if block_list is None:
        block_list = [
            [1],
            [2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21],
            [22, 23, 24, 25, 26],
            [27, 28, 29, 30, 31],
            [32],
            [32],
            [32],
            [32]
        ]
    elif len(block_list) == 0:
        raise ValueError("generate_random_map_list: 输入方块列表为空")
    elif [True if len(x) == 0 else False for x in block_list]:
        raise ValueError("generate_random_map_list: 输入方块列表中有方块无类别")

    output_list = [random.sample(x, 1)[0] for x in block_list]
    random.shuffle(output_list)
    return output_list

def calculate_difference(list1, list2, weights=(0.25, 0.35, 0.4)) -> dict:
    """
    计算两个12位置类别列表的差异值

    :param list1: 第一个列表，长度12，元素为1-32的整数（32代表空）
    :param list2: 第二个列表，长度12，元素为1-32的整数（32代表空）
    :param weights: 三个差异值的权重，默认(1/3, 1/3, 1/3)
    :return: 包含三个归一化差异值和总差异值的字典,键：
            "空/非空状态差异值": float,
            "类别差异值": float,
            "相同类别位置差异值": float,
            "总差异值": float
    """
    # -------------------------- 输入验证 --------------------------
    if len(list1) != 12 or len(list2) != 12:
        raise ValueError("两个列表的长度必须都是12")
    for lst in [list1, list2]:
        if not all(isinstance(x, int) and 1 <= x <= 32 for x in lst):
            raise ValueError("列表元素必须是1-32之间的整数")
    if sum(weights) != 1:
        raise ValueError("权重加和不为1")

    # -------------------------- 1. 空/非空状态差异值 (d1) --------------------------
    d1 = 0
    for a, b in zip(list1, list2):
        a_empty = (a == 32)
        b_empty = (b == 32)
        if a_empty != b_empty:
            d1 += 1
    # print(d1)
    # if d1 == 8:
    #     print(f"d1 = 8 : {list2}")

    # -------------------------- 2. 类别差异值 (d2) --------------------------
    # 提取非空类别集合
    set1 = {x for x in list1 if x != 32}
    set2 = {x for x in list2 if x != 32}
    # 计算对称差（只在一个集合中出现的类别数）
    symmetric_diff = set1.symmetric_difference(set2)
    d2 = len(symmetric_diff)
    # print(d2)
    # if d2 == 14:
    #     print(f"d2 = 14 : {list2}")

    # -------------------------- 3. 相同类别位置差异值 (d3) --------------------------
    # TODO: 有bug，完全相同的序列，计算出来差异值是4
    def pos_to_coord(pos):
        """将1-12的位置编号转换为地图坐标(x, y)"""
        idx = pos - 1  # 转换为0-11的索引
        x = (idx % 3) + 1  # 列号1-3
        y = (idx // 3) + 1  # 行号1-4（从下往上）
        return x, y

    d_list = []
    for m, idx1 in enumerate(list1):
        if idx1 == 32:
            continue
        elif idx1 == list2[m]:
            d_list.append(0)
            continue
        for n, idx2 in enumerate(list2):
            if idx2 == 32:
                continue
            if idx1 == idx2:
                pos1 = pos_to_coord(m + 1)
                pos2 = pos_to_coord(n + 1)
                # 计算相同类别的曼哈顿距离
                d = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                d_list.append(d)
                print(f"{m} {n} : {d}")
    if len(d_list) == 0:
        d3 = 5
    else:
        avg_d = sum(d_list) / len(d_list)
        d3 = avg_d

    # print(d_list)
    # if d3 == 5:
    #     print(f"d3 = 5 : {list2}")

    # if d1 == 8 and d2 == 14 and d3 == 5:
    #     print(f"max: {list2}")

    # # 建立类别到位置的映射（假设每个类别最多出现一次）
    # cat_pos1 = {}
    # for idx, cat in enumerate(list1):
    #     if cat != 32:
    #         cat_pos1[cat] = idx + 1  # 位置是1-12
    #
    # cat_pos2 = {}
    # for idx, cat in enumerate(list2):
    #     if cat != 32:
    #         cat_pos2[cat] = idx + 1
    #
    # # 计算相同类别的平均曼哈顿距离
    # common_cats = set(cat_pos1.keys()) & set(cat_pos2.keys())
    # if not common_cats:
    #     avg_d = 5  # 无相同类别时取最大距离5
    # else:
    #     total_d = 0
    #     for cat in common_cats:
    #         pos1 = cat_pos1[cat]
    #         pos2 = cat_pos2[cat]
    #         x1, y1 = pos_to_coord(pos1)
    #         x2, y2 = pos_to_coord(pos2)
    #         # 计算曼哈顿距离（最小格数）
    #         manhattan = abs(x1 - x2) + abs(y1 - y2)
    #         total_d += manhattan
    #     avg_d = total_d / len(common_cats)
    # d3 = avg_d
    # print(d3)

    # -------------------------- 归一化到0-100 --------------------------
    norm_d1 = (d1 / 12) * 100  # d1范围0-12
    norm_d2 = (d2 / 24) * 100  # d2范围0-24
    norm_d3 = (d3 / 5) * 100  # d3范围0-5

    # -------------------------- 加权求和 --------------------------
    w1, w2, w3 = weights
    total_diff = w1 * norm_d1 + w2 * norm_d2 + w3 * norm_d3

    # -------------------------- 返回结果 --------------------------
    return {
        "空/非空状态差异值": round(norm_d1, 2),
        "类别差异值": round(norm_d2, 2),
        "相同类别位置差异值": round(norm_d3, 2),
        "总差异值": round(total_diff, 2)
    }

def read_from_json(json_path:str) -> list:
    if not os.path.exists(json_path):
        root_path = os.path.split(json_path)[0]
        os.makedirs(root_path, exist_ok=True)
        # print(root_path)
        open(json_path, 'w').close()
    with open(json_path, 'r') as json_file:
        json_file.seek(0)
        if len(json_file.readlines()) == 0:
            return list()
        json_file.seek(0)
        data = json.load(json_file)
    return data

def write_to_json(input_map_list:list, output_json_path:str) -> None:
    root_path = os.path.split(output_json_path)[0]
    # print(root_path)
    os.makedirs(root_path, exist_ok=True)

    if not os.path.exists(output_json_path):
        open(output_json_path, 'w').close()

    with open(output_json_path, 'w') as output_file:
        output_file.write("[\n")
        for k, x in enumerate(input_map_list):
            map = x["map"]
            avg_diff = x["avg_diff"]
            data = (f"    {{\n"
                    f"        \"map\": {map},\n"
                    f"        \"avg_diff\": {avg_diff}\n"
                    f"    }}")
            if k != len(input_map_list) - 1:
                data += ",\n"
            output_file.write(data)
        output_file.write("\n]")

    # with open(output_json_path, 'r') as output_file:
    #     # json.dump(input_map_list, output_file, indent=4)
    #     output_file.seek(0)
    #     ori_data = output_file.readlines()
    #
    # try:
    #     with open(output_json_path, 'w') as output_file:
    #         # print(len(output_file.readlines()))
    #         if not len(ori_data) == 0:
    #             ori_data.pop()
    #             output_file.writelines(ori_data)
    #             output_file.write(",\n")
    #         else:
    #             output_file.write("[\n")
    #         for k, x in enumerate(input_map_list):
    #             map = x["map"]
    #             avg_diff = x["avg_diff"]
    #             data = (f"    {{\n"
    #                     f"        \"map\": {map},\n"
    #                     f"        \"avg_diff\": {avg_diff}\n"
    #                     f"    }}")
    #             if k != len(input_map_list) - 1:
    #                 data += ",\n"
    #             output_file.write(data)
    #         output_file.write("\n]")
    # except:
    #     print("ERROR")
    #     print(f"raw_data: \n"
    #           f"{ori_data}")

if __name__ == "__main__":

    # 设置种子
    random.seed(234645)   # 最好每次都修改一个值

    # 方块序列
    block_list = [
        [1],
        [2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21],
        [22, 23, 24, 25, 26],
        [27, 28, 29, 30, 31],
        [32],
        [32],
        [32],
        [32]
    ]

    # 保存地图的json路径
    save_json_path = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\output\test1.json"

    # 差异值阈值，差异值越大，地图差的越多
    difference_thres = 40

    # 设置迭代次数
    optimize_times = 10000

    # 各差异值权重
    empty_diff, label_diff, same_label_position_diff = 0.25, 0.35, 0.4

    # 可设置初始地图
    best_output = []
    # best_output.append({
    #     "map": [32, 32, 32, 28, 13, 26, 32, 5, 4, 10, 19, 1],
    #     "avg_diff": 101
    # })

    # 读取之前生成过的地图
    [best_output.append(x) for x in read_from_json(save_json_path)]
    old_map_amount = len(best_output)
    # 为区分不同批次，之前生成的地图avg_diff在原来基础上加100
    for best in best_output:
        best["avg_diff"] += 100

    # 如果json为空，且未设置初始地图，则随机生成一个
    if len(best_output) == 0:
        origin_map = generate_random_map_list(block_list)
        best_output.append({
            "map": origin_map,
            "avg_diff": 101
        })

    # 迭代生成地图
    # print(f"origin map:{best_output}")
    print("origin map:（为了区分不同批次生成的地图，avg_diff会在原来基础上加100）")
    [print(x) for x in best_output]
    print(f"总数: {old_map_amount}")
    for i in range(optimize_times):

        # 生成随机地图
        temp_map = generate_random_map_list(block_list)
        is_like = False
        difference_sum = 0

        # 遍历best_output的所有地图，计算和其他best之间的差异值
        for best in best_output:

            # 计算和其他best之间的difference，要大于一个阈值
            difference = calculate_difference(temp_map, best["map"], (empty_diff, label_diff, same_label_position_diff))
            difference_sum += difference["总差异值"]
            # print(difference["总差异值"])
            if difference["总差异值"] < difference_thres:
                is_like = True
                break
        if not is_like:
            total = len(best_output)
            best_output.append({
                "map": temp_map,
                "avg_diff": difference_sum / total
            })

    # 按avg_diff由大到小排序
    def sort_best_output(map_dict: dict):
        return map_dict["avg_diff"]
    best_output.sort(key=sort_best_output, reverse=True) # 由大到小排序

    # 打印输出生成的地图
    print("------------------------------")
    print("generate map: ")
    new_map_amount = 0
    for best in best_output:
        if best["avg_diff"] <= 100:
            print(best)
            new_map_amount += 1

    if not new_map_amount:
        print("生成了0个大于阈值的地图，请调整阈值!")
    else:
        # 询问用户是否写入json
        user_input = input(f"请输入保存至json的地图数量 -1: 全部, 0-n: 数量\n")
        input_is_legal = False
        while not input_is_legal:
            try:
                user_input = int(user_input.strip())
                if -1 <= user_input <= new_map_amount:
                    input_is_legal = True
            except ValueError as e:
                user_input = input(f"输入有误，必须是大于-1小于生成数量的整数\n")
                input_is_legal = False
        # print(user_input)

        if user_input == -1:

            write_to_json(best_output, save_json_path)
            print(f"结果已写入： {save_json_path}")
        elif user_input > 0:

            write_to_json(best_output, save_json_path)
            print(f"结果已写入： {save_json_path}")


    # difference_dict = calculate_difference([1,7,2,12,3,17,22,27,32,32,32,32], [32,32,8,13,5,18,23,28,32,4,32,1])
    # difference_dict = calculate_difference([1, 19, 10, 32, 6, 6, 32, 15, 32, 31, 25, 32], [1, 19, 10, 32, 6, 6, 32, 15, 32, 31, 25, 32])
    # print(difference_dict)
    # json_data = read_from_json(output_json_path)
    # print(json_data)

    # print(best_output)
    # for k in range(100):
    #     print(f"origin map:{origin_map}")
    #     for i in range(1000):
    #         temp_map = generate__map_list()
    #         difference_dict = calculate_difference(origin_map, temp_map)
    #         # print(temp_map)
    #         # print(difference_dict)
    #         if difference_dict["总差异值"] > max_difference:
    #             max_difference = difference_dict["总差异值"]
    #             best_map = temp_map
    #     print(max_difference)
    #     # print(best_map)
    #     origin_map = generate__map_list()
    #
    # print("------------")
    # print(max_difference)
    # print(best_map)

    # print(f"origin map:{origin_map}")
    # for i in range(10000):
    #     temp_map = generate__map_list()
    #     difference_dict = calculate_difference(origin_map, temp_map)
    #     # print(temp_map)
    #     # print(difference_dict)
    #     if difference_dict["总差异值"] >= max_difference:
    #         if len(best_output) == 0:
    #             best_output.append({
    #                 "map": temp_map,
    #                 "difference": difference_dict
    #             })
    #         else:
    #             is_like = False
    #             for best in best_output:
    #                 # 计算和其他best之间的difference，要大于一个阈值
    #                 difference = calculate_difference(temp_map, best["map"])
    #                 # print(difference["总差异值"])
    #                 if difference["总差异值"] < 45:
    #                     is_like = True
    #                     break
    #             if not is_like:
    #                 best_output.append({
    #                     "map": temp_map,
    #                     "difference": difference_dict
    #                 })
    # for best in best_output:
    #     print(best)
    # print(best_map)