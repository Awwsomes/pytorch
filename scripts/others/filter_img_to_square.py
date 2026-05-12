import cv2
import os

def pad_image_to_square(image, color):
    """
    将图片填充为正方形，原图居中，周围填充指定颜色
    :param image: 原始图片 (OpenCV Mat 对象)
    :param color: 填充颜色 (BGR 格式的元组，例如 (255, 255, 255) 代表白色)
    :return: 填充后的正方形图片
    """
    # 获取原始图片的高度和宽度
    height, width = image.shape[:2]

    # 确定正方形的边长（取长边的长度）
    side_length = max(height, width)

    # 计算上下左右需要填充的像素宽度
    # 使用 // 2 计算顶部和左侧，剩余的留给底部和右侧，以解决奇数像素的居中问题
    pad_top = (side_length - height) // 2
    pad_bottom = side_length - height - pad_top
    pad_left = (side_length - width) // 2
    pad_right = side_length - width - pad_left

    # 执行填充操作
    # cv2.BORDER_CONSTANT 表示使用常量纯色填充
    padded_image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded_image

if __name__ == "__main__":
    # ================= 配置区域 =================
    input_path = r"C:\Users\tianc\Desktop\images"       # 输入图片路径
    output_path = r"C:\Users\tianc\Desktop\images_square"     # 输出图片路径
    fill_color = (255, 0, 0)         # 填充颜色 (B, G, R)，这里默认为黑色
    # ===========================================

    os.makedirs(output_path, exist_ok=True)
    img_list = os.listdir(input_path)

    for img_name in img_list:
        old_img_path = os.path.join(input_path, img_name)
        new_img_path = os.path.join(output_path, img_name)

        # 1. 读取图片
        img = cv2.imread(old_img_path)
        if img is None:
            print(f"Error: 无法读取图片，请检查路径: {old_img_path}")
            exit()

        # 2. 处理图片（填充为正方形）
        result_img = pad_image_to_square(img, fill_color)

        # 3. 写入图片
        cv2.imwrite(new_img_path, result_img)
        print(f"处理完成，图片已保存至: {new_img_path}")