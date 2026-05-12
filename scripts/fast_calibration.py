# import napari
import numpy as np
import cv2

from calibration import solve_pnp
from calibration import compute_reprojection_error

# 写入坐标到两个文件
def write_to_txt(lidar_to_camera, reprojection_error, output_txt_path:str):
    try:
        with open(output_txt_path, 'w') as output_file:
            lines = (f"lidar_to_camera:\n"
                     f"{lidar_to_camera[0,0]}, {lidar_to_camera[0,1]}, {lidar_to_camera[0,2]}, {lidar_to_camera[0,3]},\n"
                     f"{lidar_to_camera[1,0]}, {lidar_to_camera[1,1]}, {lidar_to_camera[1,2]}, {lidar_to_camera[1,3]},\n"
                     f"{lidar_to_camera[2,0]}, {lidar_to_camera[2,1]}, {lidar_to_camera[2,2]}, {lidar_to_camera[2,3]},\n"
                     f"{lidar_to_camera[3,0]}, {lidar_to_camera[3,1]}, {lidar_to_camera[3,2]}, {lidar_to_camera[3,3]}\n")
            lines += (f"\nworld_to_camera reprojection_error: "
                      f"{reprojection_error}")
            output_file.writelines(lines)
        print(f"已写入到txt: {output_txt_path}")
    except:
        print(f"写入txt异常: {output_txt_path}")

def fast_calibration(example_img_path, img_path, world_to_lidar, points_3d, output_txt_path):
    points_2d = []  # （y,x）

    # print("--------------1.标定二维点---------------")
    # # 展示图片，处理鼠标点击事件
    # img = cv2.imread(img_path)
    # (img_height, img_width) = img.shape[:2]
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (viewer,layer) = napari.imshow(img_rgb)
    # example_img = cv2.imread(example_img_path)
    # example_img_rgb = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
    # napari.imshow(example_img_rgb)
    # # viewer.events.key_press.connect(keyboard_callback)
    #
    # # 3. 创建空的 Points 图层（关键！用于画点）
    # points_layer = viewer.add_points(
    #     name="标注点",
    #     size=8,          # 点的大小
    #     face_color="red",# 点的填充色
    #     # 初始化空的 features 表格，用于存储序号
    #     features={"index": []},
    #     text={           # 显示序号的配置
    #         "string": "{index}",  # 显示内容：序号
    #         "size": 12,
    #         "color": "blue",
    #         "anchor": "upper_left" # 序号位置
    #     },
    #     ndim=2  # 强制 2D 图层
    # )
    #
    # # 4. 【核心代码】使用 bind_key 装饰器绑定空格键
    # @viewer.bind_key('Space')
    # def keyboard_callback(viewer):
    #     point = viewer.cursor.position
    #     point = (float(point[0]), float(point[1]))
    #
    #     if 0 <= point[0] <= img_height and 0 <= point[1] <= img_width:
    #         points_2d.append(point)
    #         points_layer.add(point)
    #
    #         # --- 更新 features 序号
    #         num_points = len(points_layer.data)
    #         # 直接构建一个新的 DataFrame 赋值回去
    #         points_layer.features = {
    #             "index": np.arange(1, num_points + 1)
    #         }
    #
    #         print(f"记录第{points_2d.index(point) + 1}个坐标: {point}")
    #     else:
    #         print(f"坐标越界: {point}")
    #
    # napari.run()

    # points_2d = np.array([
    #     [998,314],
    #     [901,842],
    #     [1048,1169],
    #     [839,1198],
    #     [773,1197],
    #     [737,1455],
    #     [729,972],
    #     [763,663],
    #     [951,245]
    # ], dtype=np.float64)

    points_2d = np.array([
        [998,314],
        [901,842],
        [1048,1169],
        [839,1198],
        [773,1197],
        [737,1455],
        [729,972],
        [763,663],
        [951,245]
    ], dtype=np.float64)

    # 标定
    # 相机内参
    print("--------------2.生成lidar_to_camera矩阵---------------")
    if len(points_2d) == 0:
        print(f"标定二维点数为0！")
        return
    elif len(points_2d) != len(points_3d):
        print(f"二维点数 {len(points_2d)} 不匹配 三维点数 {len(points_3d)} !")
        return

    camera_matrix = np.array([
        # 原有内参
        [1380.4350, 0, 974.0183],
        [0, 1385.0788, 541.4301],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)

    # points_2d = np.stack(points_2d, axis=0)
    rvec,tvec = solve_pnp(points_3d, points_2d, camera_matrix, dist_coeffs)
    reprojection_error = compute_reprojection_error(points_3d, points_2d, rvec, tvec, camera_matrix, dist_coeffs)

    R, _ = cv2.Rodrigues(rvec)  # 罗德里格斯公式：向量→矩阵
    world_to_camera = np.eye(4,dtype=np.float64)
    world_to_camera[:3, :3] = R
    world_to_camera[:3, 3] = tvec.T

    lidar_to_camera = world_to_camera @ np.linalg.inv(world_to_lidar)
    # 打印
    print(f"world_to_lidar:\n{world_to_lidar}")
    print(f"world_to_camera:\n{world_to_camera}")
    print(f"reprojection_error: {reprojection_error}")
    print(f"lidar_to_camera:\n {lidar_to_camera}")

    print(f"写入到txt: {output_txt_path}")
    write_to_txt(lidar_to_camera, reprojection_error, output_txt_path)

if __name__ == "__main__":
    img_path = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\img1.png"
    output_txt_path = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\test.txt"
    world_to_lidar = np.array([[0.847164, -0.53127, -0.00805017, -1.86391],
                               [0.531232, 0.847202, -0.00643553, -0.028872],
                               [0.0102391, 0.00117545, 0.999947, -0.838359],
                               [0, 0, 0, 1]], dtype=np.float64)

    # 设置默认坐标
    points_3d = np.array([
        [4.4, - 1.2, 0.4],
        [4.4, - 2.4, 0.4],
        [3.2, - 2.4, 0.4],
        [4.4, - 3.6, 0.4],
        [4.4, - 3.6, 0.6],
        [4.4, - 4.8, 0.6],
        [5.6, - 3.6, 0.6],
        [5.6, - 2.4, 0.6],
        [5.6, - 1.2, 0.2]
    ])

    example_img_path = r"D:\A_myData\RC26-Vision\Pytorch\pytorch\scripts\test_file\img_example.png"

    fast_calibration(example_img_path, img_path, world_to_lidar, points_3d, output_txt_path)