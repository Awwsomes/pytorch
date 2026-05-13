import numpy as np
import cv2
import openvino as ov
import time
import argparse

def preprocess_image(
        image: np.ndarray,
        input_size: tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, tuple[float, int, int]]:
    """
    图像前处理：适配视频帧输入\n
    尺寸的缩放与填充，维度的改变，归一化像素数值
    核心api：\n
    # 往输入图片四周填充指定颜色的像素\n
    cv2.copyMakeBorder(src: Mat,top: int,bottom: int,left: int,
                       right: int, borderType: int,value: 颜色) -> Mat
    返回：预处理后的张量、(缩放比例, 填充宽度, 填充高度)
    """
    # 1. 尺寸缩放并填充至模型输入尺寸
    # 缩放，保持图片长宽比，避免变形
    h, w = image.shape[:2]
    target_h, target_w = input_size
    scale = min(target_w / w, target_h / h)  # 取最小的缩放因子，让图片的长边等于模型输入尺寸
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # 此时图片的长边等于模型输入尺寸

    # 填充，让图片居中，并在周围填充像素至模型输入尺寸
    pad_w, pad_h = target_w - new_w, target_h - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 往输入图片四周填充指定颜色的像素

    # 格式转换
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)  # BGR->RGB
    img_normalized = img_rgb.astype(np.float32) / 255.0  # 归一化
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC->CHW
    input_tensor = np.expand_dims(img_transposed, axis=0)  # 最前面新增batch维度

    return input_tensor, (scale, left, top)

# TODO:一对一头模式（end2end=True）具体解释，为什么就不用NMS抑制筛选重叠框
# TODO:适配没有可见度输出的模型，只有关键点的位置
def postprocess(output0, scale, pad_w, pad_h, num_key_point_list, conf_thres=0.25 ) -> tuple[np.ndarray,np.ndarray,np.ndarray,list]:
    """
    图像后处理
    目标是 拿到原始图上 高过置信度和iou阈值 的 若干个边界框和掩膜，还有对应的类别和置信度
    输出层：[1,300,4+2+max_num_kp*3] x1,y1,x2,y2,conf,label_idx,key_point1_x,key_point1_y,可见度,key_point2_x,key_point2_y,可见度...
    识别框的格式是：左上角x,y，右下角x,y
    由于每个类别的关键点数量不一样，输出形状中的max_num_kp*3是指以所有类别中关键点个数的最大值作为输出形状（若某类别没有这么多关键点，后面的无效的就会全部填0）
    :param output0: 模型原始输出
    :param scale: 前处理时图片放缩系数
    :param pad_w: 前处理时宽方向上图片需填充的像素数
    :param pad_h: 前处理时高方向上图片需填充的像素数
    :param num_key_point_list: 储存了每个类别的关键点数量，顺序按照类别映射顺序
    :param conf_thres: 置信度阈值
    :return: boxes, scores, class_ids, key_points
    """

    ## 处理边界框
    predictions = np.squeeze(output0)  # 去除为1的维度

    # 分离出边界框，置信度，类别，掩码
    # 每个锚框都只预测一个框和一个掩膜，并给出所有类别的置信度
    boxes = predictions[:, :4]
    scores = predictions[:,4] # 取出每个锚框置信度最高的那一个
    class_ids =predictions[:,5].astype(np.uint)  # 取出每个锚框置信度最高的那一个对应的类别序号
    key_points = predictions[:, 6:]  # 取出每个锚框内的掩膜

    # 置信度过滤
    mask = scores > conf_thres
    boxes, scores, class_ids, key_points = boxes[mask], scores[mask], class_ids[mask], key_points[mask]
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([]), list()

    # # NMS
    # indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    # if len(indices) == 0:
    #     return np.array([]), np.array([]), np.array([]), np.array([])
    #
    # # 增加展平操作，保证索引的安全性
    # indices = np.array(indices).flatten()
    # boxes, scores, class_ids, mask_coeffs = boxes[indices], scores[indices], class_ids[indices], mask_coeffs[indices]

    # 坐标映射回原图（缩放和填充）
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
    boxes = boxes.astype(np.int32)

    ## 处理关键点
    key_points_fix = []
    for i, key_point in enumerate(key_points):
        # 根据类别的关键点数选出有效关键点
        num_key_point = num_key_point_list[class_ids[i]]
        key_point_no_blank = key_point[:num_key_point*3]
        # 点反归一化
        for k in range(0, num_key_point*3, 3):
            key_point_no_blank[k] = (key_point_no_blank[k] - pad_w) / scale
            key_point_no_blank[k+1] = (key_point_no_blank[k+1] - pad_h) / scale
        key_points_fix.append(key_point_no_blank)

    return boxes, scores, class_ids, key_points_fix

def draw_results(img, boxes, key_points, class_ids):
    np.random.seed(42)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(80)]
    for box, key_point, class_id in zip(boxes, key_points, class_ids):
        color = colors[class_id]
        x1, y1, x2, y2 = box

        # 防止坐标越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        # 画矩形框
        # print(box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Class {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 画关键点
        for i in range(0, len(key_point), 3):
            color2 = colors[np.random.randint(0,80)]
            cv2.circle(img, (int(key_point[i]), int(key_point[i+1])), 3, color2, -1)
            cv2.putText(img, f"{key_point[i + 2]:.2f}", (int(key_point[i]), int(key_point[i+1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

    return img

def infer_single_img(img: cv2.typing.MatLike, num_key_point_list:list,
                     compiled_model: ov.CompiledModel,
                     infer_request: ov.InferRequest,
                     conf_thres=0.3, input_shape=(640,640)) -> tuple[np.ndarray,np.ndarray,np.ndarray,list]:
    """
    单张图片推理
    :param img: 输入图片
    :param num_key_point_list:
    :param compiled_model: ov模型
    :param infer_request: ov推理接口
    :return: 后处理结果 boxes, scores, class_ids, key_points
    """
    # 获取输入的形状和两层输出的形状
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # 前处理
    input_tensor, (scale, pad_left, pad_top) = preprocess_image(img, input_shape)

    # 推理
    infer_request.infer({input_layer: input_tensor})
    output = infer_request.get_tensor(output_layer).data

    # 后处理
    boxes, scores, class_ids, key_points = postprocess(output, scale, pad_left, pad_top, num_key_point_list, conf_thres)

    return boxes, scores, class_ids, key_points

def infer_video(source, num_key_point_list:list, compiled_model: ov.CompiledModel, infer_request: ov.InferRequest,
                conf_thres=0.3, input_shape=(640, 640)):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开视频流: {source}")
        return

    while True:
        success, img = cap.read()
        if not success:
            break

        start_time = time.time()
        boxes, scores, class_ids, masks = infer_single_img(img, num_key_point_list, compiled_model, infer_request, conf_thres, input_shape)
        end_time = time.time()
        infer_time = end_time - start_time
        infer_time *= 1000

        if len(boxes) > 0:
            img = draw_results(img, boxes, masks, class_ids)
            print(f"classes: {class_ids}, scores: {scores}, time: {infer_time:.2f} ms")
        else:
            print(f"No detections, time: {infer_time:.2f} ms")

        cv2.imshow("infer_result", img)

        # 修复点3：使用 ord('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=r"D:\A_myData\RC26-Vision\Pytorch\yolo26\runs\pose\train\weights\best_openvino_model\best.xml",
                        help="OpenVINO model (.xml/.onnx)")
    parser.add_argument("--source", default="1", help="source: camera id (0,1...), video or img path")
    parser.add_argument("--output", default="", help="Output image path (only for image input)")
    parser.add_argument("--num_key_point_list", default=[17], help="储存了每个类别的关键点数量，顺序按照类别映射顺序")
    parser.add_argument("--conf_thres", type=float, default=0.3)
    parser.add_argument("--input_shape", type=tuple[int,int], default=(640,640), help="模型输入尺寸")
    args = parser.parse_args()

    core = ov.Core()
    model = core.read_model(args.model)
    compiled_model = core.compile_model(model, "AUTO")
    infer_request = compiled_model.create_infer_request()

    # 简单的 source 类型判断
    source = args.source
    if source.isdigit():
        source = int(source)

    if isinstance(source, int):
        print(f"打开摄像头: {source}")
        infer_video(source, args.num_key_point_list, compiled_model,
                    infer_request, args.conf_thres, args.input_shape)
    else:
        if source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            print(f"处理图片: {source}")
            img = cv2.imread(source)
            if img is None:
                print(f"无法读取图片: {source}")
                return

            boxes, scores, class_ids, masks = infer_single_img(img, args.num_key_point_list, compiled_model, infer_request,
                                                               args.conf_thres, args.input_shape)

            if len(boxes) > 0:
                result_img = draw_results(img, boxes, masks, class_ids)
                if args.output:
                    cv2.imwrite(args.output, result_img)
                    print(f"结果已保存至: {args.output}")
                # cv2.resize(result_img, result_img, None, 0.5, 0.5)
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.imshow("Result", result_img)
                cv2.moveWindow("Result", 0, 0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No detections")
        elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"处理视频: {source}")
            infer_video(source, args.num_key_point_list, compiled_model,
                        infer_request, args.conf_thres, args.input_shape)
        else:
            print(f"不支持的文件格式: {source}")

if __name__ == "__main__":
    main()