from PIL import Image
import os


def batch_rotate_images(folder_path,output_path, save_mode="new"):
    """
    批量处理文件夹内所有图片，旋转180°并修改像素
    :param folder_path: 文件夹路径
    :param save_mode: "overwrite" 覆盖原文件，"new" 另存为新文件
    """
    # 支持的图片格式
    supported_formats = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    # 遍历文件夹
    for file_name in os.listdir(folder_path):
        # 过滤非图片文件
        if not file_name.lower().endswith(supported_formats):
            continue

        # 拼接完整路径
        image_path = os.path.join(folder_path, file_name)
        try:
            # 打开并旋转图片
            with Image.open(image_path) as img:
                rotated_img = img.rotate(180, expand=True)

                # 确定保存路径
                if save_mode == "overwrite":
                    save_path = image_path
                else:
                    name, ext = os.path.splitext(file_name)
                    save_path = os.path.join(output_path, f"{name}{ext}")

                # 保存（清空EXIF旋转标签）
                if ext.lower() in [".jpg", ".jpeg"]:
                    rotated_img.save(save_path, quality=95, exif=b"")
                else:
                    rotated_img.save(save_path)

                print(f"处理完成：{file_name} -> {os.path.basename(save_path)}")

        except Exception as e:
            print(f"处理失败 {file_name}：{str(e)}")


# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 替换为你的图片文件夹路径（绝对路径/相对路径都可以）
    your_folder_path = "D:\A_myData\dataset\camera_calibrate\calibrate2_0119"
    output_path = r"D:\A_myData\dataset\camera_calibrate\calibrate3_0119"

    # 批量处理（推荐用new模式，避免覆盖原文件）
    batch_rotate_images(your_folder_path,output_path, save_mode="new")