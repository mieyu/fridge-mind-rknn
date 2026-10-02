import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


DEFAULT_MODEL_FP = "/home/ztl/fridge_project/model/best_fp.rknn"
DEFAULT_MODEL_INT8 = "/home/ztl/fridge_project/model/best.rknn"
DEFAULT_CLASSES = "/home/ztl/fridge_project/model/classes.txt"

CAMERA_DEVICE = "/dev/video18"

CAP_WIDTH = 640
CAP_HEIGHT = 480
CAP_FPS = 30

IMG_SIZE = 416
CONF_THRES = 0.25
IOU_THRES = 0.45

# RK3566 上不要每帧都推理，先每 5 帧推理一次
INFER_INTERVAL = 5


def choose_default_model():
    if Path(DEFAULT_MODEL_FP).exists():
        return DEFAULT_MODEL_FP
    return DEFAULT_MODEL_INT8


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


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
        return pred.T.astype(np.float32)

    raise RuntimeError(f"unexpected output shape: {pred.shape}, expected (?, 11)")


def postprocess(pred, classes, conf_thres, scale, pad_w, pad_h, orig_w, orig_h):
    boxes_xywh = pred[:, 0:4].astype(np.float32)
    obj_conf = pred[:, 4].astype(np.float32)
    cls_scores = pred[:, 5:].astype(np.float32)

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
    obj_conf = obj_conf[keep]
    cls_conf = cls_conf[keep]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

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


def draw_results(frame, results, infer_ms, fps):
    vis = frame.copy()

    for det in results:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name']} {det['score']:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        y_text = max(20, y1 - 6)
        cv2.putText(
            vis,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        vis,
        f"FPS:{fps:.1f} Infer:{infer_ms:.1f}ms",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        vis,
        "Press q to quit",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    return vis


def open_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera: {device}")

    # 必须强制 MJPG，避免 YUYV 高带宽/高CPU
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

    # 尽量降低缓存延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print("[INFO] camera opened:", device)
    print("[INFO] width:", real_w)
    print("[INFO] height:", real_h)
    print("[INFO] fps:", real_fps)
    print("[INFO] fourcc:", fourcc_str)

    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=choose_default_model())
    parser.add_argument("--classes", default=DEFAULT_CLASSES)
    parser.add_argument("--camera", default=CAMERA_DEVICE)
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--interval", type=int, default=INFER_INTERVAL)
    parser.add_argument("--mode", choices=["RGB", "BGR"], default="RGB")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)

    if not Path(args.classes).exists():
        raise FileNotFoundError(args.classes)

    classes = load_classes(args.classes)

    print("[INFO] classes:", classes)
    print("[INFO] model:", args.model)
    print("[INFO] camera:", args.camera)
    print("[INFO] conf:", args.conf)
    print("[INFO] infer interval:", args.interval)
    print("[INFO] input mode:", args.mode)

    rknn = RKNNLite()

    print("[INFO] loading RKNN model...")
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    print("[INFO] init runtime...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("[ERROR] init_runtime failed:", ret)
        print("[HINT] try: sudo chmod 666 /dev/rknpu")
        rknn.release()
        return

    cap = open_camera(args.camera)

    frame_id = 0
    last_results = []
    infer_ms = 0.0

    last_time = time.time()
    fps = 0.0

    cv2.namedWindow("RK3566 Fridge Camera", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame_bgr = cap.read()

            if not ret or frame_bgr is None:
                print("[WARN] failed to read frame")
                time.sleep(0.02)
                continue

            frame_id += 1
            orig_h, orig_w = frame_bgr.shape[:2]

            now = time.time()
            dt = now - last_time
            last_time = now

            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if frame_id % args.interval == 0:
                if args.mode == "RGB":
                    input_src = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    input_src = frame_bgr

                img_input, scale, pad_w, pad_h = letterbox(input_src, IMG_SIZE)
                input_data = np.expand_dims(img_input, axis=0).astype(np.uint8)

                t0 = time.time()
                outputs = rknn.inference(inputs=[input_data])
                infer_ms = (time.time() - t0) * 1000.0

                if outputs is not None:
                    pred = normalize_output(outputs[0])

                    obj_max = float(np.max(pred[:, 4]))
                    cls_max = float(np.max(pred[:, 5:]))

                    if obj_max == 0.0 and cls_max == 0.0:
                        last_results = []
                        print("[WARN] obj and cls all zero, model output abnormal")
                    else:
                        last_results = postprocess(
                            pred=pred,
                            classes=classes,
                            conf_thres=args.conf,
                            scale=scale,
                            pad_w=pad_w,
                            pad_h=pad_h,
                            orig_w=orig_w,
                            orig_h=orig_h,
                        )

                        if last_results:
                            msg = []
                            for det in last_results:
                                msg.append(f"{det['class_name']}:{det['score']:.2f}")
                            print("[DETECT]", ", ".join(msg))

            vis = draw_results(frame_bgr, last_results, infer_ms, fps)

            cv2.imshow("RK3566 Fridge Camera", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        rknn.release()
        cv2.destroyAllWindows()
        print("[INFO] camera test stopped")


if __name__ == "__main__":
    main()