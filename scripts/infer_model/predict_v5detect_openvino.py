import cv2
import numpy as np
import openvino.runtime as ov
import time
from pathlib import Path

class YOLOv5OpenVINO:
    def __init__(self, model_path:str, classes:list, device:str="CPU", conf_thres:float=0.3, iou_thres:float=0.3):
        """
        初始化YOLOv5 OpenVINO推理器

        Args:
            model_path: OpenVINO模型XML文件路径或模型目录
            device: 推理设备 ("CPU", "GPU", "AUTO", "MULTI:CPU,GPU"等)
            conf_thres: 置信度阈值
            iou_thres: NMS的IOU阈值
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes

        # 初始化OpenVINO核心
        self.core = ov.Core()

        # 读取模型
        if Path(model_path).is_dir():
            # 如果是目录，自动查找xml文件
            xml_files = list(Path(model_path).glob("*.xml"))
            if not xml_files:
                raise FileNotFoundError(f"在目录 {model_path} 中未找到XML模型文件")
            model_path = str(xml_files[0])

        self.model = self.core.read_model(model=model_path)

        # 编译模型
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device)

        # 获取输入输出层信息
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # 获取输入尺寸
        self.input_shape = self.input_layer.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(f"模型加载成功")
        print(f"输入尺寸: {self.input_width}x{self.input_height}")
        print(f"推理设备: {device}")

    def preprocess(self, image):
        """
        YOLOv5标准前处理：等比例缩放+填充+归一化+通道转换

        Args:
            image: OpenCV读取的BGR图像 (H, W, 3)

        Returns:
            blob: 预处理后的输入张量 (1, 3, H, W)
            ratio: 缩放比例
            pad: 填充大小 (dw, dh)
        """
        # 原始图像尺寸
        h, w = image.shape[:2]

        # 计算缩放比例
        r = min(self.input_height / h, self.input_width / w)

        # 计算缩放后的尺寸
        new_unpad = (int(round(w * r)), int(round(h * r)))

        # 计算需要填充的大小
        dw, dh = self.input_width - new_unpad[0], self.input_height - new_unpad[1]

        # 将填充分到两边
        dw /= 2
        dh /= 2

        # 缩放图像
        if (w, h) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 添加填充
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换为RGB格式并归一化
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0

        # 添加batch维度
        blob = np.expand_dims(image, axis=0)

        return blob, r, (dw, dh)

    def postprocess(self, outputs, ratio, pad):
        """
        YOLOv5标准后处理：解码输出+非极大值抑制

        Args:
            outputs: 模型原始输出 (1, 25200, 85)
            ratio: 图像缩放比例
            pad: 填充大小 (dw, dh)

        Returns:
            boxes: 检测框坐标 (x1, y1, x2, y2)
            scores: 置信度分数
            class_ids: 类别ID
        """
        # 移除batch维度
        outputs = outputs[0]

        # 过滤置信度低的检测
        mask = outputs[:, 4] > self.conf_thres
        outputs = outputs[mask]

        if len(outputs) == 0:
            return np.array([]), np.array([]), np.array([])

        # 计算类别概率
        outputs[:, 5:] *= outputs[:, 4:5]

        # 获取每个检测的最大类别概率和对应的类别ID
        class_ids = np.argmax(outputs[:, 5:], axis=1)
        scores = np.max(outputs[:, 5:], axis=1)

        # 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx = outputs[:, 0]
        cy = outputs[:, 1]
        w = outputs[:, 2]
        h = outputs[:, 3]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # 调整边界框坐标（去除填充和缩放）
        dw, dh = pad
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        boxes = np.column_stack((x1, y1, x2, y2))

        # 非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)

        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        # 提取NMS后的结果
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]

        return boxes, scores, class_ids

    def draw_detections(self, image, boxes, scores, class_ids):
        """
        在图像上绘制检测结果

        Args:
            image: 原始图像
            boxes: 检测框坐标
            scores: 置信度分数
            class_ids: 类别ID

        Returns:
            image: 绘制了检测结果的图像
        """
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签和置信度
            label = f"{self.classes[class_id]}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

            cv2.rectangle(image, (x1, label_y - label_size[1] - 5),
                          (x1 + label_size[0], label_y + 5), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def infer_image(self, image_path, save_path=None, show=False):
        """
        推理单张图片

        Args:
            image_path: 图片路径
            save_path: 结果保存路径，为None则不保存
            show: 是否显示结果

        Returns:
            result_image: 检测结果图像
            inference_time: 推理时间(ms)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 前处理
        start_time = time.time()
        blob, ratio, pad = self.preprocess(image)
        preprocess_time = (time.time() - start_time) * 1000

        # 推理
        start_time = time.time()
        outputs = self.compiled_model([blob])[self.output_layer]
        inference_time = (time.time() - start_time) * 1000

        # 后处理
        start_time = time.time()
        boxes, scores, class_ids = self.postprocess(outputs, ratio, pad)
        postprocess_time = (time.time() - start_time) * 1000

        # 绘制结果
        result_image = self.draw_detections(image.copy(), boxes, scores, class_ids)

        # 打印性能信息
        print(f"\n单张图片推理结果:")
        print(f"检测到 {len(boxes)} 个目标")
        print(f"前处理时间: {preprocess_time:.2f} ms")
        print(f"推理时间: {inference_time:.2f} ms")
        print(f"后处理时间: {postprocess_time:.2f} ms")
        print(f"总时间: {preprocess_time + inference_time + postprocess_time:.2f} ms")

        # 保存结果
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"结果已保存到: {save_path}")

        # 显示结果
        if show:
            cv2.imshow("Detection Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result_image, inference_time

    def infer_video(self, video_source, save_path=None, show=True, fps=30):
        """
        推理视频流（支持本地视频文件和摄像头）

        Args:
            video_source: 视频源，可以是文件路径或摄像头ID(如0)
            save_path: 结果保存路径，为None则不保存
            show: 是否实时显示结果
            fps: 保存视频的帧率
        """
        # 打开视频流
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {video_source}")

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n视频流推理开始:")
        print(f"视频尺寸: {width}x{height}")
        print(f"总帧数: {total_frames}" if total_frames > 0 else "摄像头模式")

        # 初始化视频写入器
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"结果将保存到: {save_path}")

        # 性能统计
        total_inference_time = 0
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 前处理
                blob, ratio, pad = self.preprocess(frame)

                # 推理
                start_time = time.time()
                outputs = self.compiled_model([blob])[self.output_layer]
                inference_time = (time.time() - start_time) * 1000

                # 后处理
                boxes, scores, class_ids = self.postprocess(outputs, ratio, pad)

                # 绘制结果
                result_frame = self.draw_detections(frame, boxes, scores, class_ids)

                # 绘制FPS信息
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                cv2.putText(result_frame, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 保存帧
                if writer:
                    writer.write(result_frame)

                # 显示帧
                if show:
                    cv2.imshow("YOLOv5 OpenVINO Detection", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断推理")
                        break

                # 更新统计
                total_inference_time += inference_time
                frame_count += 1

                # 打印进度
                if frame_count % 10 == 0:
                    avg_fps = 1000 * frame_count / total_inference_time
                    print(f"已处理 {frame_count} 帧, 平均FPS: {avg_fps:.1f}", end='\r')

        finally:
            # 释放资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            # 打印最终统计
            if frame_count > 0:
                avg_inference_time = total_inference_time / frame_count
                avg_fps = 1000 / avg_inference_time
                print(f"\n\n视频流推理完成:")
                print(f"总处理帧数: {frame_count}")
                print(f"平均推理时间: {avg_inference_time:.2f} ms")
                print(f"平均FPS: {avg_fps:.1f}")


# 使用示例
if __name__ == "__main__":
    # 模型路径（替换为你的OpenVINO模型路径）
    MODEL_PATH = r"D:\A_myData\RC26-Vision\Pytorch\yolov5-master\runs\train\卷轴检测_han2\weights\卷轴检测_han2\best.xml"

    # 初始化推理器
    detector = YOLOv5OpenVINO(
        model_path=MODEL_PATH,
        classes=["blue"],
        device="CPU",  # 可以改为 "GPU", "AUTO", "MULTI:CPU,GPU" 等
        conf_thres=0.8,
        iou_thres=0.3
    )

    # # 1. 推理单张图片
    # print("\n=== 单张图片推理 ===")
    # try:
    #     detector.infer_image(
    #         image_path=r"C:\Users\tianc\Desktop\img5.jpg",  # 替换为你的图片路径
    #         save_path="",
    #         show=True
    #     )
    # except FileNotFoundError as e:
    #     print(f"图片推理跳过: {e}")

    # 2. 推理视频流
    print("\n=== 视频流推理 ===")
    try:
        # 推理本地视频文件
        # detector.infer_video(
        #     video_source="test.mp4",  # 替换为你的视频路径
        #     save_path="result.mp4",
        #     show=True
        # )

        # 推理摄像头（按q退出）
        detector.infer_video(
            video_source=1,  # 摄像头ID
            show=True
        )
    except Exception as e:
        print(f"视频推理错误: {e}")