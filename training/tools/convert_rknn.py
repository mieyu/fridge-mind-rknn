"""将单输出 ONNX 转换为 RK3566 的浮点或 INT8 RKNN 模型。"""

import argparse
from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ROOT / "training/data/datasets/fridge_food_v1/classes.txt"
IMAGES = ROOT / "training/data/datasets/fridge_food_v1/images/train"


def prepare_calibration(images_dir, output_dir, count):
    """按板端的 416 letterbox 方式生成量化校准图片列表。"""
    import cv2

    images = sorted(p for p in images_dir.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    and not p.name.startswith("._"))
    if not images:
        raise ValueError(f"没有找到校准图片：{images_dir}")
    random.Random(0).shuffle(images)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, path in enumerate(images[:count]):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取校准图片：{path}")
        height, width = image.shape[:2]
        scale = min(416 / width, 416 / height)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        image = cv2.resize(image, (new_w, new_h))
        left, top = (416 - new_w) // 2, (416 - new_h) // 2
        image = cv2.copyMakeBorder(image, top, 416 - new_h - top,
                                   left, 416 - new_w - left,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        target = output_dir / f"{index:04d}.png"
        # 用标准图片文件，由 Toolkit 按 RGB 读取；不在这里除以 255。
        if not cv2.imwrite(str(target), image):
            raise RuntimeError(f"校准图片写入失败：{target}")
        paths.append(str(target.resolve()))
    dataset = output_dir / "dataset.txt"
    dataset.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return dataset


def require_success(code, stage):
    if code != 0:
        raise RuntimeError(f"{stage} 失败，RKNN 返回码：{code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "training/artifacts/deploy")
    parser.add_argument("--classes", type=Path, default=CLASSES)
    parser.add_argument("--int8", action="store_true", help="默认生成浮点模型；指定后生成 INT8")
    parser.add_argument("--calibration-images", type=Path, default=IMAGES)
    parser.add_argument("--calibration-count", type=int, default=100)
    args = parser.parse_args()
    onnx = args.onnx.resolve()
    output_dir = args.output_dir.resolve()
    target = output_dir / ("best.rknn" if args.int8 else "best_fp.rknn")
    if not onnx.is_file():
        parser.error("--onnx 文件不存在")
    if target.exists():
        parser.error("目标模型已存在，请指定新的 --output-dir")
    if args.calibration_count <= 0:
        parser.error("--calibration-count 必须为正数")
    classes = args.classes.read_text(encoding="utf-8-sig").split()
    if classes != ["apple", "banana", "milk_box", "coke_can", "egg", "blueberry_box"]:
        parser.error("classes.txt 必须与现有板端的 6 类顺序一致")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = None
    if args.int8:
        dataset = prepare_calibration(args.calibration_images.resolve(),
                                      output_dir / "calibration", args.calibration_count)

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        require_success(rknn.config(mean_values=[[0, 0, 0]],
                                    std_values=[[255, 255, 255]],
                                    target_platform="rk3566"), "config")
        require_success(rknn.load_onnx(model=str(onnx)), "load_onnx")
        build_options = {"do_quantization": args.int8}
        if dataset is not None:
            build_options["dataset"] = str(dataset)
        require_success(rknn.build(**build_options), "build")
        require_success(rknn.export_rknn(str(target)), "export_rknn")
    finally:
        rknn.release()

    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    packaged_onnx = output_dir / "best.onnx"
    if onnx != packaged_onnx and not packaged_onnx.exists():
        shutil.copy2(onnx, packaged_onnx)
    print(f"RKNN：{target}")


if __name__ == "__main__":
    main()
