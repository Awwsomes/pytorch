import cv2
import numpy as np
from openvino import Core


def main():
    # 1. 初始化 OpenVINO Core
    core = Core()

    model_path = r"D:\A_myData\RC26-Vision\Pytorch\yolo26\runs\obb\obb1_6800_去年的数据集\weights\best_openvino_model_op13\model.xml"
    source = 1
    conf_threshold = 0.3
    iou_threshold = 0.3
    infer_device = "CPU"

    print("正在加载和编译模型...")
    # 读取模型
    model = core.read_model(model=model_path)

    # 编译模型。Windows 上如果安装了 Intel 显卡驱动，可以将 "CPU" 改为 "GPU" 以加速
    compiled_model = core.compile_model(model=model, device_name=infer_device)

    # 获取输入和输出层
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # 获取模型需要的输入形状 (通常格式为 [N, C, H, W])
    # 动态 batch size 时，N 可能为 -1，这里假设 N=1
    N, C, H, W = input_layer.shape

    # 2. 打开视频流
    # 传入 0 表示使用默认摄像头，也可传入视频路径如 "test_video.mp4"
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("错误：无法打开视频流。")
        return

    print("开始推理，按 'q' 键退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ==========================================
        # 3. 图像预处理
        # ==========================================
        # resized_frame = cv2.resize(frame, (W, H))
        # 等比缩放
        orig_h, orig_w = frame.shape[:2]
        scale_w, scale_h = W / orig_w , H / orig_h
        scale = int(min(scale_w, scale_h))
        new_w, new_h = orig_w * scale , orig_h * scale
        resized_frame = cv2.resize(frame, (new_w, new_h))
        # 计算四周填充像素量
        pad_w, pad_h = abs(W - new_w), abs(H - new_h)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        top, down = pad_h // 2, pad_h - (pad_h // 2)
        resized_frame = cv2.copyMakeBorder(resized_frame, top, down, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 灰色，不会和环境产生太大对比而被当成特征

        # cv2.imshow("resize", resized_frame)
        # if cv2.waitKey(1) == 'q':
        #     break
        # continue

        input_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)   # astype:将数组中所有元素转换成指定类型
        input_image /= 255.0  # 归一化像素数值

        # ==========================================
        # 4. 执行推理
        # ==========================================
        results = compiled_model([input_image])[output_layer]  # 推理返回的对象类似字典，可以用需要的输出层名字作为键索引，也可以通过compiled_model.output(0)获取接口
        # results = compiled_model([input_image])[output_layer.names.pop()]
        # print(results.shape)

        # ==========================================
        # 5. 后处理与可视化
        # ==========================================
        # [1,300,7]: x,y,w,h,conf,class_idx,angle(弧度)
        detections = results[0]  # [300, 7]

        # [调试用] 如果屏幕没框，看控制台打印的最大置信度是多少
        max_conf = np.max(detections[:, 4])
        if max_conf > 0.1:  # 只有置信度大于 0.1 时才打印，避免刷屏
            print(f"Max confidence in frame: {max_conf:.4f}")

        boxes, scores, class_ids = [], [], []

        # 筛选出高于置信度的旋转框
        for i in range(detections.shape[0]):
            # 7个维度: [x, y, w, h, score, class_id, angle]
            x, y, w, h, score, class_id, angle = detections[i]

            # 置信度筛选
            if score > conf_threshold:
                # 映射坐标
                cx = (x - left) / scale
                cy = (y - top) / scale
                bw = w / scale
                bh = h / scale

                angle_deg = angle * 180.0 / np.pi

                boxes.append(((float(cx), float(cy)), (float(bw), float(bh)), float(angle_deg)))
                scores.append(float(score))
                class_ids.append(int(class_id))

        # iou重合框筛选
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxesRotated(boxes, scores, conf_threshold, iou_threshold)

            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                scores = [scores[i] for i in indices]
                class_ids = [class_ids[i] for i in indices]
            else:
                boxes = []
                scores = []
                class_ids = []

        # 旋转框可视化
        for i in range(len(boxes)):
            box = boxes[i]
            # 获取顶点并绘制
            pts = cv2.boxPoints(box)
            pts = np.int32(pts)
            cv2.drawContours(frame, [pts], 0, (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{class_ids[i]} {scores[i]:.2f}",
                        (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            print(f"idx: {class_ids[i]} score: {scores[i]} angle: {box[2]}")

        cv2.imshow("OpenVINO YOLO26-OBB", frame)
        ret = cv2.waitKey(1)
        if ret == 113 or ret == 81:
            break

    # 6. 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()