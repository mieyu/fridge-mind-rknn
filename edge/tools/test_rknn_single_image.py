import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


DEFAULT_MODEL_FP = "/home/ztl/fridge_project/model/best_fp.rknn"
DEFAULT_MODEL_INT8 = "/home/ztl/fridge_project/model/best.rknn"
DEFAULT_CLASSES = "/home/ztl/fridge_project/model/classes.txt"
DEFAULT_IMAGE = "/home/ztl/fridge_project/test_images/test.jpg"
DEFAULT_SAVE = "/home/ztl/fridge_project/test_images/rknn_result.jpg"

IMG_SIZE = 416
IOU_THRES = 0.45


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes


def letterbox(img, new_size=416, color=(114, 114, 114)):
    h, w = img.shape[:2]

    scale = min(new_size / w, new_size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)

    pad_w = (new_size - new_w) // 2
    pad_h = (new_size - new_h) // 2

    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return canvas, scale, pad_w, pad_h


def xywh_to_xyxy(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def normalize_output(output):
    pred = np.squeeze(output)

    if pred.ndim != 2:
        raise RuntimeError(f"unexpected output ndim: {pred.ndim}, shape={pred.shape}")

    if pred.shape[1] == 11:
        return pred.astype(np.float32)

    if pred.shape[0] == 11:
        print("[WARN] output is transposed, using pred.T")
        return pred.T.astype(np.float32)

    raise RuntimeError(f"unexpected output shape: {pred.shape}, expected (?, 11)")


def print_debug(pred, classes, topk=20):
    boxes = pred[:, 0:4]
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]

    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = np.max(cls_scores, axis=1)

    scores = obj * cls_conf

    print("------------------------------------------------------------")
    print("[DEBUG] pred shape:", pred.shape)
    print("[DEBUG] pred min/max:", float(np.min(pred)), float(np.max(pred)))
    print("[DEBUG] boxes min/max:", float(np.min(boxes)), float(np.max(boxes)))
    print("[DEBUG] obj min/max:", float(np.min(obj)), float(np.max(obj)))
    print("[DEBUG] cls min/max:", float(np.min(cls_scores)), float(np.max(cls_scores)))
    print("[DEBUG] score min/max:", float(np.min(scores)), float(np.max(scores)))

    top_idx = np.argsort(scores)[-topk:][::-1]

    print(f"[DEBUG] top {topk} candidates by obj*cls:")
    for idx in top_idx:
        cid = int(cls_ids[idx])
        cname = classes[cid] if 0 <= cid < len(classes) else f"class_{cid}"
        print(
            "idx={:<6d} score={:.6f} obj={:.6f} cls_conf={:.6f} cls={} box={}".format(
                int(idx),
                float(scores[idx]),
                float(obj[idx]),
                float(cls_conf[idx]),
                cname,
                [round(float(x), 3) for x in boxes[idx].tolist()],
            )
        )

    return scores


def postprocess(pred, classes, conf_thres, scale, pad_w, pad_h, orig_w, orig_h):
    boxes_xywh = pred[:, 0:4].astype(np.float32)
    obj_conf = pred[:, 4].astype(np.float32)
    cls_scores = pred[:, 5:].astype(np.float32)

    # 当前 RKNN 输出应该已经是 decoded bbox：x,y,w,h,obj,classes
    # 如果 box 是 0~1 归一化，则恢复到 416 尺度
    if np.max(boxes_xywh) <= 2.0:
        boxes_xywh *= IMG_SIZE

    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = np.max(cls_scores, axis=1)
    scores = obj_conf * cls_conf

    keep = scores > conf_thres

    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    # letterbox 416 坐标还原到原图
    boxes_xyxy[:, [0, 2]] -= pad_w
    boxes_xyxy[:, [1, 3]] -= pad_h
    boxes_xyxy[:, :4] /= scale

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

    nms_boxes = []
    valid_xyxy = []
    valid_scores = []
    valid_cls_ids = []

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        if w < 2 or h < 2:
            continue

        nms_boxes.append([int(x1), int(y1), int(w), int(h)])
        valid_xyxy.append([int(x1), int(y1), int(x2), int(y2)])
        valid_scores.append(float(scores[i]))
        valid_cls_ids.append(int(cls_ids[i]))

    if len(nms_boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=valid_scores,
        score_threshold=conf_thres,
        nms_threshold=IOU_THRES,
    )

    if len(indices) == 0:
        return []

    indices = np.array(indices).reshape(-1)

    results = []

    for i in indices:
        cid = valid_cls_ids[i]
        cname = classes[cid] if 0 <= cid < len(classes) else f"class_{cid}"

        results.append(
            {
                "class_id": cid,
                "class_name": cname,
                "score": valid_scores[i],
                "box": valid_xyxy[i],
            }
        )

    return results


def draw_results(img_bgr, results, save_path):
    vis = img_bgr.copy()

    for det in results:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name']} {det['score']:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(save_path, vis)
    print("[INFO] result image saved:", save_path)


def choose_default_model():
    if Path(DEFAULT_MODEL_FP).exists():
        return DEFAULT_MODEL_FP
    return DEFAULT_MODEL_INT8


def run_inference(model_path, classes_path, image_path, save_path, conf_thres, input_mode):
    for p in [model_path, classes_path, image_path]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    classes = load_classes(classes_path)
    print("[INFO] classes:", classes)
    print("[INFO] model:", model_path)
    print("[INFO] image:", image_path)
    print("[INFO] conf_thres:", conf_thres)
    print("[INFO] input_mode:", input_mode)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    print("[INFO] original image shape:", img_bgr.shape)

    if input_mode == "RGB":
        img_input_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif input_mode == "BGR":
        img_input_src = img_bgr
    else:
        raise ValueError("input_mode must be RGB or BGR")

    img_input, scale, pad_w, pad_h = letterbox(img_input_src, IMG_SIZE)

    # RKNNLite Python 接口这里使用 NHWC uint8
    input_data = np.expand_dims(img_input, axis=0).astype(np.uint8)

    print("[INFO] input shape:", input_data.shape, input_data.dtype)

    rknn = RKNNLite()

    print("[INFO] loading RKNN model...")
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    print("[INFO] init runtime...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("[ERROR] init_runtime failed:", ret)
        print("[HINT] try: sudo chmod 666 /dev/rknpu")
        rknn.release()
        return

    print("[INFO] running inference...")
    outputs = rknn.inference(inputs=[input_data])

    if outputs is None:
        rknn.release()
        raise RuntimeError("inference failed: outputs is None")

    print("[INFO] output count:", len(outputs))
    for i, out in enumerate(outputs):
        print(f"[INFO] output[{i}] shape={out.shape}, dtype={out.dtype}")

    pred = normalize_output(outputs[0])

    print_debug(pred, classes, topk=20)

    obj_max = float(np.max(pred[:, 4]))
    cls_max = float(np.max(pred[:, 5:]))

    if obj_max == 0.0 and cls_max == 0.0:
        print("============================================================")
        print("[DIAGNOSE] obj_conf and cls_scores are all zero.")
        print("[DIAGNOSE] This RKNN model has abnormal confidence/class output.")
        print("[DIAGNOSE] If this is best.rknn INT8, please test best_fp.rknn.")
        print("[DIAGNOSE] If best_fp.rknn is also zero, re-export ONNX/RKNN is needed.")
        print("============================================================")
        rknn.release()
        return

    results = postprocess(
        pred=pred,
        classes=classes,
        conf_thres=conf_thres,
        scale=scale,
        pad_w=pad_w,
        pad_h=pad_h,
        orig_w=orig_w,
        orig_h=orig_h,
    )

    print("============================================================")
    print("[RESULT] detections:")
    if len(results) == 0:
        print("No object detected.")
    else:
        for det in results:
            print(
                f"{det['class_name']:15s} "
                f"score={det['score']:.6f} "
                f"box={det['box']}"
            )
    print("============================================================")

    draw_results(img_bgr, results, save_path)

    rknn.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=choose_default_model())
    parser.add_argument("--classes", default=DEFAULT_CLASSES)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--save", default=DEFAULT_SAVE)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mode", choices=["RGB", "BGR"], default="RGB")
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        classes_path=args.classes,
        image_path=args.image,
        save_path=args.save,
        conf_thres=args.conf,
        input_mode=args.mode,
    )


if __name__ == "__main__":
    main()