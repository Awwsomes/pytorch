import warnings
import cv2
import numpy as np
import json
import os
from tqdm import tqdm


def solve_pnp(object_points, image_points, camera_matrix, dist_coeffs):
    """
    核心函数：求解旋转向量和平移向量
    :param object_points: 人为输入的3D世界点 (N×3 numpy数组)
    :param image_points: 人为输入的2D像素点 (N×2 numpy数组)
    :param camera_matrix: 相机内参矩阵 (3×3 numpy数组)
    :param dist_coeffs: 相机畸变系数 (1×5 数组，无畸变则全零)
    :return: rvec (旋转向量), tvec (平移向量)
    """
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE  # 迭代法，适配多点点优化，精度更高
    )

    if not success:
        raise RuntimeError("PnP求解失败！请检查：1.点数量≥4 2.点非共面 3.世界点与像素点一一对应")

    return rvec, tvec


def compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """辅助函数：计算平均重投影误差（验证结果精度，越小越好）"""
    # 将3D世界点通过求解的位姿投影回2D图像
    projected_2d, _ = cv2.projectPoints(
        objectPoints=object_points,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    projected_2d = projected_2d.reshape(-1, 2)  # 格式转换为[N×2]

    # 计算每个点的投影误差（欧氏距离），返回平均值
    point_errors = np.linalg.norm(image_points - projected_2d, axis=1)
    return np.mean(point_errors)

def read_image_points(json_path:str):
    """
    读取像素坐标
    :param json_path: json路径
    :return: 返回(n,2)的numpy矩阵
    """
    with open(json_path,'r') as json_file:
        json_data = json.load(json_file)
        shapes = json_data["shapes"]
        amount_points = len(shapes)
        points = np.zeros((amount_points,2))
        for idx,shape in enumerate(shapes):
            point = shape["points"][0]
            points[idx,:] = point[:]
    return points

def read_object_points(txt_path:str):
    """
    读取三维坐标点
    :param txt_path: txt路径
    :return: (n,3)的numpy矩阵
    """
    with open(txt_path,'r') as txt:
        lines = txt.readlines()
        amount_point = len(lines)
        points = np.zeros((amount_point,3),dtype=float)
        for idx,line in enumerate(lines):
            line = line.strip().split()
            point = np.zeros((1,3))
            for index,num in enumerate(line):
                if index > 3:
                    warnings.warn(f"{txt_path} line{idx + 1} 3d point has value more than 3,skip ...")
                    break
                point[0,index] = float(num)
            # print(point[0,:])
            points[idx,:] = point[0,:]
    return points

def write_rt_txt(output_txt_path,rvec,tvec,reprojection_error):
    """
    将旋转向量，平移向量，RT矩阵，平均重投影误差写入txt
    :param output_txt_path:  输出txt路径
    :param rvec:
    :param tvec:
    :param reprojection_error:
    :return: none

    Note：不能有同名的txt文件，不会覆盖，而是报错
    """
    with open(output_txt_path,'x') as txt:
        lines = []

        lines.append(f"rvec: {rvec.T}\n")
        lines.append(f"tvec: {tvec.T}\n")

        R, _ = cv2.Rodrigues(rvec)  # 罗德里格斯公式：向量→矩阵
        lines.append(f"\n RT:\n")
        lines.append(f"{R[0,0]} {R[0,1]} {R[0,2]} {tvec[0].item()}\n")
        lines.append(f"{R[1,0]} {R[1,1]} {R[1,2]} {tvec[1].item()}\n")
        lines.append(f"{R[2,0]} {R[2,1]} {R[2,2]} {tvec[2].item()}\n")
        lines.append(f"0 0 0 1\n")

        lines.append(f"\nreprojection_error: {reprojection_error}")

        txt.writelines(lines)

def calibration(root_path,
                camera_matrix = np.array([[1380.4350, 0, 974.0183],
                                                 [0, 1385.0788, 541.4301],
                                                 [0, 0, 1]],
                                         dtype=np.float32),
                dist_coeffs = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]],
                                       dtype=np.float32)):
    """
    批量生成世界坐标系到相机坐标系的RT矩阵，输出到RT文件夹下

    :param root_path: 存放路径，全部txt，json放在一个文件夹底下
    :param camera_matrix: 相机内参
    :param dist_coeffs: 畸变系数
    :return: none

    所有文件都放在一个文件夹内即可
    对应的txt与json需同名
    """

    # 筛选txt文件
    list_txt = os.listdir(root_path)
    list_txt = [x for x in list_txt if os.path.isfile(os.path.join(root_path,x))]
    list_txt = [x for x in list_txt if x.endswith(".txt")]

    for txt_name in tqdm(list_txt):
        # 拼接路径
        root_name,_ = os.path.splitext(txt_name)
        json_name = f"{root_name}.json"
        txt_path = os.path.join(root_path,txt_name)
        json_path = os.path.join(root_path,json_name)

        # 找与txt同名的json
        if not os.path.exists(json_path):
            warnings.warn(f"{root_name} 's json not exist, skip...")
            continue

        # 读取像素坐标点
        image_points = read_image_points(json_path)

        # 读取3d坐标点
        object_points = read_object_points(txt_path)

        # print(image_points)
        # print(object_points)

        # 计算RT矩阵和重投影误差
        rvec,tvec = solve_pnp(object_points, image_points, camera_matrix, dist_coeffs)
        reprojection_error = compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)

        # 输出结果
        output_txt_name = f"{root_name}_RT.txt"
        output_path = os.path.join(root_path,"RT",output_txt_name)
        # print(output_path)
        write_rt_txt(output_path, rvec, tvec, reprojection_error)

if __name__ == "__main__":
    # 相机内参
    camera_matrix = np.array([
        # 原有内参
        [1380.4350, 0, 974.0183],
        [0, 1385.0788, 541.4301],
        [0, 0, 1]

        # [1372.91003377705, 0, 974.886023374686],
        # [0, 1374.66068396083, 549.075277677568],
        # [0, 0, 1]
    ], dtype=np.float32)

    # 相机畸变系数：[k1, k2, p1, p2, k3]（无畸变则全填0）
    # k1/k2/k3：径向畸变；p1/p2：切向畸变（从标定结果获取）
    dist_coeffs = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    # dist_coeffs = np.array([[-0.0383938069100753],[1.68404077315427],[0],[0],[0]], dtype=np.float32)

    root_dir = r"D:\A_myData\dataset\camera_calibrate\output\260202"
    calibration(root_dir)

    # # ----------------------
    # # 2. 12对3D世界点（手动测量，单位：米/毫米，需统一）
    # # 要求：① 至少4个点 ② 所有点不共面 ③ 与下方像素点一一对应（顺序完全一致）
    # # ----------------------
    # object_points = np.array([
    #     # [3.6 - 4.5999945443330965 , 4.37 - 0.4000300843703803 ,0.6 - 0.04656820724087374],  # 世界点1：【替换】实际3D坐标（如：(0,0,0)米）
    #     # [4.8 - 4.5999945443330965 , 4.37  - 0.4000300843703803,0.6 - 0.04656820724087374],  # 世界点2：【替换】实际3D坐标
    #     # [4.8 - 4.5999945443330965, 3.17 - 0.4000300843703803 ,0.0 - 0.04656820724087374],  # 世界点3：【替换】实际3D坐标
    #     # [3.6 - 4.5999945443330965,3.17 - 0.4000300843703803 ,0.0 - 0.04656820724087374],  # 世界点4：【替换】实际3D坐标
    #     # [4.37 - 0.4000300843703803, 4.5999945443330965 - 3.6, 0.6 - 0.04656820724087374],  # 世界点1：【替换】实际3D坐标（如：(0,0,0)米）
    #     # [4.37  - 0.4000300843703803, 4.5999945443330965 - 4.8, 0.6 - 0.04656820724087374],  # 世界点2：【替换】实际3D坐标
    #     # [3.17 - 0.4000300843703803, 4.5999945443330965 - 4.8, 0.0 - 0.04656820724087374],  # 世界点3：【替换】实际3D坐标
    #     # [3.17 - 0.4000300843703803, 4.5999945443330965 - 3.6, 0.0 - 0.04656820724087374],  # 世界点4：【替换】实际3D坐标
    #     # [-4.5999945443330965, 1.271181, 0.586511],  # 世界点5：【替换】实际3D坐标
    #     # [2.666630, 0.336640, 0.494546],  # 世界点6：【替换】实际3D坐标
    #     # [2.639372, 0.399690, -0.337642],  # 世界点7：【替换】实际3D坐标
    #     # [1.993336, 1.334230, -0.245677],  # 世界点8：【替换】实际3D坐标
    #     # [2.755196, 1.352697, 0.610707],  # 世界点9：【替换】实际3D坐标
    #     # [3.106076, 0.256299, 0.624341],  # 世界点10：【替换】实际3D坐标
    #     # [3.057917, 0.230999, -0.170882],  # 世界点11：【替换】实际3D坐标
    #     # [2.707038, 1.327397, -0.184515]  # 世界点12：【替换】实际3D坐标
    #
    #     # [0, 0, 0.4],
    #     # [0, 0, 0],
    #     # [1.2, 0, 0.4],
    #     # [0, 1.2, 0.6],
    #     # [1.2, 1.2, 0.6],
    #     # [2.4, 1.2, 0.4],
    #     # [2.4, 0, 0.4],
    #     # [3.6, 0, 0.4],
    #     # [1.2, 1.2, 0.4]
    #
    #     # [3.2, -4.8, 0.4],
    #     # [4.4, -4.8, 0.6],
    #     # [3.2, -3.6, 0.4],
    #     # [3.2, -2.4, 0.4],
    #     # [3.2, -1.2, 0.4],
    #     # [4.4, -3.6, 0.6],
    #     # [3.2, -1.2, 0],
    #     # [5.6, -2.4, 0.6]
    #
    #     [4.4, -3.6, 0.6],
    #     [5.6, -2.4, 0.6],
    #     [5.6, -1.2, 0.4],
    #     [4.4, -3.6, 0.4],
    #     [5.6, -3.6, 0.6],
    #     [5.6, -2.4, 0.4],
    #     [4.4, -2.4, 0.4],
    #     [5.6, -3.6, 0.4]
    #
    # ], dtype=np.float32)
    #
    # # ----------------------
    # # 3. 12对2D像素点（手动读取，单位：像素，与世界点一一对应）
    # # 读取方式：用图像查看工具（如PS、画图）获取点的(x,y)，注意：图像坐标系原点在左上角
    # # ----------------------
    # image_points = np.array([
    #     # [706.2962962962963, 717.037037037037],  # 像素点1：【替换】对应“世界点1”的像素坐标
    #     # [1009.0890688259109, 717.0040485829959],  # 像素点2：【替换】对应“世界点2”的像素坐标
    #     # [1032.072072072072, 1010.2102102102102],  # 像素点3：【替换】对应“世界点3”的像素坐标
    #     # [598.1081081081081, 1010.2702702702702],  # 像素点4：【替换】对应“世界点4”的像素坐标
    #     # [116.25000000000009, 52.88461538461538],  # 像素点5：【替换】对应“世界点5”的像素坐标
    #     # [721.0576923076923, 75.0],  # 像素点6：【替换】对应“世界点6”的像素坐标
    #     # [730.6730769230769, 450.96153846153845],  # 像素点7：【替换】对应“世界点7”的像素坐标（示例值需改）
    #     # [183.5576923076924, 528.8461538461538],  # 像素点8：【替换】对应“世界点8”的像素坐标（示例值需改）
    #     # [279.1304347826087, 13.043478260869565],  # 像素点9：【替换】对应“世界点9”的像素坐标（示例值需改）
    #     # [781.7391304347825, 26.95652173913043],  # 像素点10：【替换】对应“世界点10”的像素坐标（示例值需改）
    #     # [799.9999999999999, 362.6086956521739],  # 像素点11：【替换】对应“世界点11”的像素坐标（示例值需改）
    #     # [325.2173913043478, 386.08695652173907]  # 像素点12：【替换】对应“世界点12”的像素坐标（示例值需改）
    #
    #     # [1627.0553935860057, 766.7638483965014],
    #     # [1603.9,906.8],
    #     # [1165.8,729.5],
    #     # [1485.1077586206898,595.9051724137931],
    #     # [1136.3338788870703,569.5417348608838],
    #     # [790.8101571946796,596.5900846432891],
    #     # [714.4169611307422,692.5795053003534],
    #     # [267.8378378378378,656.2162162162161],
    #     # [1130.4974271012006,624.5283018867924]
    #
    #     # [
    #     #     1671.804008908686,
    #     #     953.674832962138
    #     # ],
    #     # [
    #     #     1649.0,
    #     #     749.3
    #     # ],
    #     # [
    #     #     1067.9331941544885,
    #     #     875.0521920668058
    #     # ],
    #     # [
    #     #     628.5714285714286,
    #     #     819.4908062234795
    #     # ],
    #     # [
    #     #     293.6,
    #     #     777.3
    #     # ],
    #     # [
    #     #     1203.3,
    #     #     716.5
    #     # ],
    #     # [
    #     #     292.1974965229486,
    #     #     910.4311543810848
    #     # ],
    #     # [
    #     #     994.1970310391363,
    #     #     655.8569500674763
    #     # ]
    #
    #     [
    #         1222.9960317460316,
    #         685.9126984126984
    #     ],
    #     [
    #         720.0222717149222,
    #         602.2271714922049
    #     ],
    #     [
    #         296.91211401425176,
    #         678.8361045130641
    #     ],
    #     [
    #         1223.3898305084745,
    #         783.2203389830509
    #     ],
    #     [
    #         1139.298669891173,
    #         591.7533252720679
    #     ],
    #     [
    #         722.4193548387096,
    #         669.516129032258
    #     ],
    #     [
    #         620.2591792656589,
    #         797.1922246220303
    #     ],
    #     [
    #         1139.193899782135,
    #         660.3485838779957
    #     ]
    #
    # ], dtype=np.float32)
    #
    # # ==============================================
    # # 以下是【无需修改】的求解与输出逻辑
    # # ==============================================
    # try:
    #     # 求解位姿（旋转向量+平移向量）
    #     rvec, tvec = solve_pnp(object_points, image_points, camera_matrix, dist_coeffs)
    #
    #     # 计算重投影误差（验证精度）
    #     avg_repro_error = compute_reprojection_error(
    #         object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
    #     )
    #
    #     # 打印结果
    #     print("=" * 50)
    #     print("               PnP位姿估计结果")
    #     print("=" * 50)
    #     print(f"1. 旋转向量（rvec）：单位=弧度")
    #     print(f"   x方向：{rvec[0][0]:.6f}")
    #     print(f"   y方向：{rvec[1][0]:.6f}")
    #     print(f"   z方向：{rvec[2][0]:.6f}")
    #     print("\n2. 平移向量（tvec）：单位=与世界点一致（如米）")
    #     print(f"   x方向：{tvec[0][0]:.6f}")
    #     print(f"   y方向：{tvec[1][0]:.6f}")
    #     print(f"   z方向：{tvec[2][0]:.6f}")
    #     print("\n3. 旋转矩阵（R）：由旋转向量转换而来")
    #     R, _ = cv2.Rodrigues(rvec)  # 罗德里格斯公式：向量→矩阵
    #     print(f"   {R[0]}")
    #     print(f"   {R[1]}")
    #     print(f"   {R[2]}")
    #     print(f"\n4. 平均重投影误差：{avg_repro_error:.4f} 像素（建议<2像素）")
    #     print("=" * 50)
    #
    # except Exception as e:
    #     print(f"\n求解出错：{str(e)}")