import openvino as ov
import cv2
import numpy as np
import argparse
import time

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

def postprocess(output0, output1, scale, pad_w, pad_h,
                img_shape,num_classes:int, conf_thres=0.25, iou_thres=0.45,
                num_masks=32,mask_thres=0.5 ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    图像后处理
    目标是 拿到原始图上 高过置信度和iou阈值 的 若干个边界框和掩膜，还有对应的类别和置信度
    :param output0: 输入一 [1,4+num_classes+32,8400]：[batch, xywh+num_classes+mask系数(一般是32个),锚框数]
    :param output1: 输入二 [1,32,160,160]: [batch,mask系数个数(一般是32个),全局共享模板形状掩膜（尺寸是模型输入尺寸的1/4）]
    :param scale: 前处理时图片放缩系数
    :param pad_w: 前处理时宽方向上图片需填充的像素数
    :param pad_h: 前处理时高方向上图片需填充的像素数
    :param img_shape: 原图尺寸
    :param num_classes: 类别个数
    :param conf_thres: 置信度阈值
    :param iou_thres: iou阈值
    :param num_masks: 掩膜系数个数（全局模板形状掩膜张数）
    :param mask_thres: 掩膜前后景阈值
    :return: boxes, scores, class_ids, masks
    """

    ## 处理边界框
    # [1,4+num_classes+32,8400] -> [8400, 4+num_classes+1]
    predictions = np.squeeze(output0).T  # 去除为1的维度并转置

    # 分离出边界框，置信度，类别，掩码
    # 每个锚框都只预测一个框和一个掩膜，并给出所有类别的置信度
    box_pred = predictions[:, :4]
    scores = np.max(predictions[:, 4:4 + num_classes], axis=1)  # 取出每个锚框置信度最高的那一个
    class_ids = np.argmax(predictions[:, 4:4 + num_classes], axis=1)  # 取出每个锚框置信度最高的那一个对应的类别序号
    mask_coeffs = predictions[:, 4 + num_classes:]  # 取出每个锚框内的掩膜

    # 置信度过滤
    mask = scores > conf_thres
    box_pred, scores, class_ids, mask_coeffs = box_pred[mask], scores[mask], class_ids[mask], mask_coeffs[mask]

    if len(box_pred) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 重叠框过滤，NMS极大值抑制
    # xywh -> xyxy
    boxes = np.zeros_like(box_pred)
    boxes[:, 0] = box_pred[:, 0] - box_pred[:, 2] / 2
    boxes[:, 1] = box_pred[:, 1] - box_pred[:, 3] / 2
    boxes[:, 2] = box_pred[:, 0] + box_pred[:, 2] / 2
    boxes[:, 3] = box_pred[:, 1] + box_pred[:, 3] / 2

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)  # TODO:为什么要给置信度阈值？
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 增加展平操作，保证索引的安全性
    indices = np.array(indices).flatten()
    boxes, scores, class_ids, mask_coeffs = boxes[indices], scores[indices], class_ids[indices], mask_coeffs[indices]

    # 坐标映射回原图（缩放和填充）
    img_h, img_w = img_shape
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
    boxes = boxes.astype(np.int32)  # TODO

    ## 处理掩膜
    # 计算每个识别框自己的掩膜
    output1 = np.squeeze(output1)  # 去除batch维度，[1,32,160,160] -> [32,160,160]
    # 所有目标共享一组模板形状（模板掩膜），通过和自己的掩膜系数加权求和，就能得到自己的掩膜
    # 一组模板一般是32张，160*160（模型输入尺寸的1/4），就是output1里的[32,160,160]
    masks = mask_coeffs @ output1.reshape(num_masks, -1)  # [32,160,160]->[32,160*160]，为了能矩阵相乘
    masks = 1 / (1 + np.exp(-masks))  # Sigmoid激活，把数值归一化，代表每个像素是前景（要识别的目标）的概率
    masks = masks.reshape(-1, 160, 160)  # 再重新拼回一张图

    # 所有掩膜放缩回原图尺寸
    mh, mw = masks.shape[1], masks.shape[2]
    masks_resized = []
    for i, mask in enumerate(masks):
        # 1. 放缩回网络输入尺寸 (160 -> 640)
        mask_net = cv2.resize(mask, (mw * 4, mh * 4))

        # 2. 提取原图对应的有效区域内的掩膜（由于输入的图不全是原图，还有填充的灰色色块）
        top, left = int(pad_h), int(pad_w)
        bottom, right = int(pad_h + img_h * scale), int(pad_w + img_w * scale)
        mask_valid = mask_net[top:bottom, left:right]

        # 3. Resize 回原图尺寸
        mask_orig = cv2.resize(mask_valid, (img_w, img_h))

        # 4. 根据 BBox 截断 Mask（消除框外的 Sigmoid 噪点）
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        mask_cropped = np.zeros_like(mask_orig)  # mask_orig形状的全零矩阵
        if x2 > x1 and y2 > y1:
            mask_cropped[y1:y2, x1:x2] = mask_orig[y1:y2, x1:x2]

        # 5. 前后景筛选（阈值为0.5）
        mask_cropped = cv2.threshold(mask_cropped, mask_thres, 255, type=np.uint8)

        masks_resized.append(mask_cropped)

    return boxes, scores, class_ids, np.array(masks_resized)

def draw_results(img, boxes, masks, class_ids):
    np.random.seed(42)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(80)]
    for box, mask, class_id in zip(boxes, masks, class_ids):
        color = colors[class_id]
        x1, y1, x2, y2 = box

        # 防止坐标越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        # 仅利用 numpy 的布尔索引在局部区域做像素混合，大幅提升绘制性能并避免背景变暗
        img[mask] = (img[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Class {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def infer_single_img(img: cv2.typing.MatLike, num_classes:int,
                     compiled_model: ov.CompiledModel,
                     infer_request: ov.InferRequest,
                     conf_thres=0.3, iou_thres=0.3,
                     num_masks=32, mask_thres=0.5,
                     input_shape=(640,640)) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    单张图片推理
    :param img: 输入图片
    :param num_classes: 类别数
    :param compiled_model: ov模型
    :param infer_request: ov推理接口
    :return: 后处理结果 boxes, scores, class_ids, masks
    """
    # 获取输入的形状和两层输出的形状
    input_layer = compiled_model.input(0)
    output0_layer = compiled_model.output(0)
    output1_layer = compiled_model.output(1)

    img_shape = img.shape[:2]

    # 前处理
    input_tensor, (scale, pad_left, pad_top) = preprocess_image(img, input_shape)

    # 推理
    # 修复点1：使用字典传入输入张量
    infer_request.infer({input_layer: input_tensor})
    output0 = infer_request.get_tensor(output0_layer).data
    output1 = infer_request.get_tensor(output1_layer).data

    # 后处理
    boxes, scores, class_ids, masks = postprocess(output0, output1, scale, pad_left, pad_top, img_shape, num_classes, conf_thres, iou_thres, num_masks, mask_thres)

    return boxes, scores, class_ids, masks

def infer_video(source, num_classes:int, compiled_model: ov.CompiledModel, infer_request: ov.InferRequest,
                conf_thres=0.3, iou_thres=0.3,num_masks=32, mask_thres=0.5,input_shape=(640, 640)):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开视频流: {source}")
        return

    while True:
        success, img = cap.read()
        if not success:
            break

        start_time = time.time()
        boxes, scores, class_ids, masks = infer_single_img(img, num_classes, compiled_model, infer_request, conf_thres, iou_thres, num_masks, mask_thres, input_shape)
        end_time = time.time()
        infer_time = end_time - start_time

        if len(boxes) > 0:
            img = draw_results(img, boxes, masks, class_ids)
            print(f"classes: {class_ids}, scores: {scores}, time: {infer_time:.3f} s")
        else:
            print(f"No detections, time: {infer_time:.3f} s")

        cv2.imshow("infer_result", img)

        # 修复点3：使用 ord('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=r"D:\A_myData\RC26-Vision\Pytorch\yolo11\runs\segment\train\weights\best_openvino_model\best.xml",
                        help="OpenVINO model (.xml/.onnx)")
    parser.add_argument("--source", default='1', help="source: camera id (0,1...), video or img path")
    parser.add_argument("--output", default="", help="Output image path (only for image input)")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes in your dataset")
    parser.add_argument("--conf_thres", type=float, default=0.3)
    parser.add_argument("--iou_thres", type=float, default=0.3)
    parser.add_argument("--input_shape", type=tuple[int,int], default=(640,640), help="模型输入尺寸")
    parser.add_argument("--num_masks", type=int, default=32, help="掩膜系数，一般YOLO固定为32，不需要调整")
    parser.add_argument("--mask_thres", type=float, default=0.5, help="非必要不要调整！模型输出的掩膜是一张概率地图，大于阈值的像素是前景即要识别的目标，小于阈值的是背景。")
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
        infer_video(source, args.num_classes, compiled_model, infer_request, args.conf_thres,
                    args.iou_thres, args.num_masks, args.mask_thres, args.input_shape)
    else:
        if source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            print(f"处理图片: {source}")
            img = cv2.imread(source)
            if img is None:
                print(f"无法读取图片: {source}")
                return

            boxes, scores, class_ids, masks = infer_single_img(img, args.num_classes, compiled_model, infer_request,
                                                               args.conf_thres, args.iou_thres, args.num_masks,
                                                               args.mask_thres, args.input_shape)

            if len(boxes) > 0:
                result_img = draw_results(img, boxes, masks, class_ids)
                if args.output:
                    cv2.imwrite(args.output, result_img)
                    print(f"结果已保存至: {args.output}")
                # cv2.resize(result_img, result_img, None, 0.5, 0.5)
                cv2.imshow("Result", result_img)
                cv2.moveWindow("Result", 0, 0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No detections")
        elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"处理视频: {source}")
            infer_video(source, args.num_classes, compiled_model, infer_request, args.conf_thres,
                        args.iou_thres, args.num_masks, args.mask_thres, args.input_shape)
        else:
            print(f"不支持的文件格式: {source}")


if __name__ == "__main__":
    main()