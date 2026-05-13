import openvino as ov
import cv2
import numpy as np
import argparse
from typing import Tuple, List
import time

np.set_printoptions(
    suppress=True,      # 禁用科学计数法
    precision=6,       # 保留6位小数（可自行修改）
    floatmode='fixed'  # 固定小数格式
)

def preprocess_image(
    image: np.ndarray,
    input_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, Tuple[float, int, int]]:
    """
    图像预处理：适配视频帧输入
    返回：预处理后的张量、(缩放比例, 填充宽度, 填充高度)
    """
    h, w = image.shape[:2]
    target_h, target_w = input_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 缩放 + 填充
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = target_w - new_w, target_h - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 格式转换
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(img_transposed, axis=0)

    return input_tensor, (scale, left, top)


def postprocess_output(
    output: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    num_classes: int = 15
) -> List[dict]:
    """后处理：旋转框解析 + NMS"""
    # print(output.shape)
    predictions = np.squeeze(output).T

    # 【新增】关键修复：如果没有任何预测结果，直接返回空列表
    # print(predictions.shape)
    # print(predictions[0,:])
    if len(predictions) == 0:
        return []

    # 解析OBB参数
    x = predictions[:, 0]
    y = predictions[:, 1]
    w = predictions[:, 2]
    h = predictions[:, 3]
    class_confs = predictions[:, 4:4+num_classes]
    angle = predictions[:, 5]

    # 置信度计算
    # print(class_confs)
    class_ids = np.argmax(class_confs, axis=1)
    class_scores = class_confs[np.arange(len(class_confs)), class_ids]
    final_scores = class_scores

    # 低置信度过滤
    mask = final_scores > conf_threshold
    x, y, w, h = x[mask], y[mask], w[mask], h[mask]
    angle, final_scores, class_ids = angle[mask], final_scores[mask], class_ids[mask]

    # 坐标映射回原图
    x_ori = (x - pad_left) / scale
    y_ori = (y - pad_top) / scale
    w_ori = w / scale
    h_ori = h / scale
    angle_deg = np.rad2deg(angle)

    # 旋转框格式
    rotated_boxes = [((x_ori[i], y_ori[i]), (w_ori[i], h_ori[i]), angle_deg[i]) for i in range(len(x_ori))]

    # 旋转框NMS
    indices = cv2.dnn.NMSBoxesRotated(rotated_boxes, final_scores, conf_threshold, nms_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "bbox": rotated_boxes[i],
                "score": final_scores[i],
                "class_id": class_ids[i]
            })
    return results


def draw_results(
    image: np.ndarray,
    results: List[dict],
    class_names: List[str] = None
) -> np.ndarray:
    """绘制旋转检测框"""
    if class_names is None:
        class_names = [
            "plane", "ship", "storage tank", "baseball diamond", "tennis court",
            "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
            "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"
        ]

    for res in results:
        box_points = cv2.boxPoints(res["bbox"])
        box_points = np.int32(box_points)
        cv2.drawContours(image, [box_points], 0, (0, 255, 0), 2)

        label = f"{class_names[res['class_id']]}: {res['score']:.2f}"
        x, y = box_points[0][0], box_points[0][1]
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        idx = res["class_id"]
        conf = res["score"]
        angle = res["bbox"][2]
        print(f"idx: {idx} conf: {conf} angle: {angle}")
    return image


def process_video(
    model_path: str,
    video_source: str,
    input_size: Tuple[int, int],
    conf_threshold: float,
    nms_threshold: float,
    num_classes: int,
    output_path: str = None,
    device: str = "CPU"
):
    """视频流处理主函数（支持视频/摄像头/RTSP）"""
    # 1. 加载OpenVINO模型
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    infer_request = compiled_model.create_infer_request()

    # 2. 打开视频流
    # 视频源：0=摄像头，rtsp地址=网络流，文件路径=本地视频
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频源: {video_source}")

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: 分辨率={width}x{height}, FPS={fps}")

    # 视频保存器
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 3. 逐帧推理循环
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束或断开

        # 预处理
        input_tensor, (scale, pad_left, pad_top) = preprocess_image(frame, input_size)

        # 推理
        output = infer_request.infer(input_tensor)[compiled_model.output(0)]

        # 后处理
        results = postprocess_output(
            output, scale, pad_left, pad_top, conf_threshold, nms_threshold, num_classes
        )


        # 绘制结果
        frame = draw_results(frame, results)

        # 实时FPS显示
        frame_count += 1
        current_fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示
        cv2.imshow("YOLOv8-OBB OpenVINO 实时检测", frame)

        # 保存视频
        if video_writer:
            video_writer.write(frame)

        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("视频处理完成！")

# TODO: 输出层格式：1 8400 4+num_classes+1 (x_c y_c w h conf_1 conf_2 ... conf_n angle)

def main():
    parser = argparse.ArgumentParser(description="YOLOv8-OBB OpenVINO 视频/图片/摄像头部署")
    parser.add_argument("--model", default=r"D:\A_myData\RC26-Vision\Pytorch\yolov8\runs\obb\train\weights\best_openvino_model\best.xml", help="OpenVINO模型路径 .xml / .onnx")
    parser.add_argument("--source", default=1, help="输入源：图片/视频路径/0(摄像头)/rtsp流地址")
    parser.add_argument("--input-size", nargs=2, type=int, default=[640, 640], help="模型输入尺寸")
    parser.add_argument("--conf", type=float, default=0.75, help="置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS阈值")
    parser.add_argument("--classes", type=int, default=1, help="类别数量")
    parser.add_argument("--output", help="输出视频/图片保存路径")
    parser.add_argument("--device", default="CPU", help="推理设备 CPU/GPU/AUTO/NPU")

    args = parser.parse_args()

    # 判断是图片还是视频
    if type(args.source) == int:
        # 处理视频/摄像头
        process_video(
            model_path=args.model,
            video_source=args.source,
            input_size=tuple(args.input_size),
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            num_classes=args.classes,
            output_path=args.output,
            device=args.device
        )
    else:
        # 处理图片
        img = cv2.imread(args.source)
        input_tensor, (scale, left, top) = preprocess_image(img, tuple(args.input_size))
        core = ov.Core()
        compiled_model = core.compile_model(core.read_model(args.model), args.device)
        output = compiled_model.infer(input_tensor)[compiled_model.output(0)]
        results = postprocess_output(output, scale, left, top, args.conf, args.nms, args.classes)
        img = draw_results(img, results)
        cv2.imwrite(args.output or "result.jpg", img)
        print(f"图片结果已保存：{args.output or 'result.jpg'}")



if __name__ == "__main__":
    main()