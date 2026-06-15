# import av
# import os
#
#
# def extract_key_frames(video_path, output_dir):
#     """
#     提取MP4视频的关键帧（I帧）并保存为JPG图片
#
#     参数:
#         video_path (str): 输入视频路径（如"input.mp4"）
#         output_dir (str): 输出图片的目录（如"key_frames"）
#     """
#     # 创建输出目录（若不存在）
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 打开视频文件
#     with av.open(video_path) as container:
#         # 定位视频流（跳过音频/字幕流）
#         video_stream = next(stream for stream in container.streams if stream.type == 'video')
#
#         # 遍历视频包（packet），解码为帧（frame）
#         for packet in container.demux(video_stream):
#             # 跳过无数据的包
#             if not packet.size:
#                 continue
#
#             # 解码包为原始帧
#             for frame in packet.decode():
#                 # 检查是否为关键帧（I帧）
#                 if frame.pict_type.name == 'I':  # 'I'表示关键帧
#                     # 生成输出路径（使用pts作为文件名，避免重复）
#                     output_path = os.path.join(output_dir, f"key_frame_{frame.pts or 'unknown'}.jpg")
#
#                     # 将帧转换为PIL图像并保存为JPG
#                     frame.to_image().save(output_path)
#                     print(f"已保存关键帧: {output_path}")
#
#
# if __name__ == "__main__":
#     extract_key_frames("input.mp4", "key_frames")


import av
import os
from PIL import Image  # 确保已安装Pillow库（pip install pillow）
from av.video.frame import PictureType

def extract_key_frames(video_path, output_dir, target_size=None, keep_aspect_ratio=False):
    """

    提取MP4视频的关键帧（I帧）并保存为JPG图片，支持调整图片大小

    参数:
        video_path (str): 输入视频路径（如"input.mp4"）
        output_dir (str): 输出图片的目录（如"key_frames"）
        target_size (tuple, optional): 目标图片尺寸（宽度, 高度），如(640, 480)。默认None表示保持原尺寸
        keep_aspect_ratio (bool, optional): 是否保持宽高比（仅当target_size有效时生效）。默认False
    """
    # 创建输出目录（若不存在）
    print(f"实际接收的视频路径：{video_path}")  # 添加此行
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 9069
    # 打开视频文件
    with av.open(video_path) as container:
        # 定位视频流（跳过音频/字幕流）
        video_stream = next(stream for stream in container.streams if stream.type == 'video')

        # 遍历视频包（packet），解码为帧（frame）
        for packet in container.demux(video_stream):
            # 跳过无数据的包
            if not packet.size:
                continue

            # 解码包为原始帧
            for frame in packet.decode():
                # 检查是否为关键帧（I帧）
                #if frame.pict_type.name == 'I':  # 'I'表示关键帧
                if frame.pict_type == PictureType.I:  # 直接比较整数值（0）
                    # 生成输出路径（使用pts作为文件名，避免重复）
                    frame_count += 1
                    #output_path = os.path.join(output_dir, f"key_frame_{frame.pts or 'unknown'}.jpg")
                    output_path = os.path.join(output_dir, f"juanZhou{frame_count}.png")
                    # 将帧转换为PIL图像
                    image = frame.to_image()

                    # 调整图片大小（如果指定了target_size）
                    if target_size is not None:
                        # 校验target_size格式
                        if not (isinstance(target_size, tuple) and len(target_size) == 2 and
                                all(isinstance(dim, int) and dim > 0 for dim in target_size)):
                            raise ValueError("target_size必须为包含两个正整数的元组（如(640, 480)）")

                        # 计算保持宽高比的目标尺寸（可选）
                        original_width, original_height = image.size
                        target_width, target_height = target_size

                        if keep_aspect_ratio:
                            # 计算宽高比缩放后的尺寸（以宽度或高度为基准）
                            ratio = min(target_width / original_width, target_height / original_height)
                            new_width = int(original_width * ratio)
                            new_height = int(original_height * ratio)
                            target_size = (new_width, new_height)
                            print(f"保持宽高比，实际输出尺寸：{new_width}x{new_height}")

                        # 调整大小（使用高质量滤波器LANCZOS）
                        image = image.resize(target_size, resample=Image.Resampling.LANCZOS)

                    # 保存为JPG（可修改为.png等其他格式）
                    image.save(output_path)
                    print(f"已保存关键帧: {output_path}")


if __name__ == "__main__":
    # 示例1：不调整尺寸（保持原尺寸）
    extract_key_frames(r"D:\A_myData\dataset\rawVideo\lack100.mp4", r"D:\\A_myData\\dataset\\juanZhou5")

    # 示例2：调整为目标尺寸（640x480，不保持宽高比）
    # extract_key_frames("D:\\Pytorch\\yolov5-master\\src\\data\\video\\red5.mp4", r"D:/Pytorch/yolov5-master/src/data/images", target_size=(1920, 1080))

    # 示例3：调整为目标尺寸（640x480，保持宽高比）
    # extract_key_frames("input.mp4", "key_frames", target_size=(640, 480), keep_aspect_ratio=True)


# import av
# import os
# from PIL import Image
#
#
# def extract_key_frames(video_path, output_dir, target_size=None, keep_aspect_ratio=False):
#     """
#     提取MP4视频的关键帧（I帧）并保存为JPG图片，支持调整图片大小
#
#     参数:
#         video_path (str): 输入视频路径（如"input.mp4"）
#         output_dir (str): 输出图片的目录（如"key_frames"）
#         target_size (tuple, optional): 目标图片尺寸（宽度, 高度），如(640, 480)。默认None表示保持原尺寸
#         keep_aspect_ratio (bool, optional): 是否保持宽高比（仅当target_size有效时生效）。默认False
#     """
#     # 创建输出目录（若不存在）
#     os.makedirs(output_dir, exist_ok=True)
#     frame_count = 0
#
#     # 打开视频文件
#     try:
#         with av.open(video_path) as container:
#             # 收集所有视频流（可能有多个，但通常取第一个）
#             video_streams = [stream for stream in container.streams if stream.type == 'video']
#
#             # 检查是否存在视频流
#             if not video_streams:
#                 raise ValueError(f"视频文件 '{video_path}' 中未检测到视频流（可能是纯音频文件或损坏）")
#
#             # 选择第一个视频流（常见场景）
#             video_stream = video_streams[0]
#             print(f"找到视频流，编码格式: {video_stream.codec_context.codec.name}")
#
#             # 遍历视频包（packet），解码为帧（frame）
#             for packet in container.demux(video_stream):
#                 # 跳过无数据的包
#                 if not packet.size:
#                     continue
#
#                 # 解码包为原始帧
#                 for frame in packet.decode():
#                     # 检查是否为关键帧（I帧）
#                     if frame.pict_type.name == 'I':  # 'I'表示关键帧
#                         frame_count += 1
#                         output_path = os.path.join(output_dir, f"key_frame_{frame_count}.jpg")
#
#                         # 将帧转换为PIL图像
#                         image = frame.to_image()
#
#                         # 调整图片大小（如果指定了target_size）
#                         if target_size is not None:
#                             # 校验target_size格式
#                             if not (isinstance(target_size, tuple) and len(target_size) == 2 and
#                                     all(isinstance(dim, int) and dim > 0 for dim in target_size)):
#                                 raise ValueError("target_size必须为包含两个正整数的元组（如(640, 480)）")
#
#                             # 计算保持宽高比的目标尺寸（可选）
#                             original_width, original_height = image.size
#                             target_width, target_height = target_size
#
#                             if keep_aspect_ratio:
#                                 # 计算宽高比缩放后的尺寸（以宽度或高度为基准）
#                                 ratio = min(target_width / original_width, target_height / original_height)
#                                 new_width = int(original_width * ratio)
#                                 new_height = int(original_height * ratio)
#                                 target_size = (new_width, new_height)
#                                 print(f"保持宽高比，实际输出尺寸：{new_width}x{new_height}")
#
#                             # 调整大小（使用高质量滤波器LANCZOS）
#                             image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
#
#                         # 保存为JPG（可修改为.png等其他格式）
#                         image.save(output_path)
#                         print(f"已保存关键帧: {output_path}")
#
#     except av.AVError as e:
#         raise RuntimeError(f"无法打开视频文件 '{video_path}'，可能是文件损坏或格式不支持: {e}")
#     except Exception as e:
#         raise RuntimeError(f"处理视频时发生错误: {e}")
#
#
# if __name__ == "__main__":
#     # 示例：提取关键帧并调整为1920x1080（不保持宽高比）
#     try:
#         extract_key_frames(
#             video_path="D:/Pytorch/yolov5-master/src/data/video/red.mp4",
#             output_dir=r"D:/Pytorch/yolov5-master/src/data/images",
#             target_size=(1920, 1080)
#         )
#     except Exception as e:
#         print(f"程序执行失败: {e}")

