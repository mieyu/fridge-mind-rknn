"""使用外部 YOLOv5 源码训练冰箱食材模型。"""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training/data/datasets/fridge_food_v1/fridge_food.yaml"
HYP = ROOT / "training/experiments/fridge_yolov5n_416/hyp.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolov5-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--dataset-root", type=Path,
                        help="默认使用 data 配置所在目录，忽略配置中旧的绝对 path")
    parser.add_argument("--weights", default="yolov5n.pt")
    parser.add_argument("--hyp", type=Path, default=HYP)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="", help="如 0 或 cpu；空值由 YOLOv5 自动选择")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "training/artifacts/fridge_yolov5n_416")
    args = parser.parse_args()

    repo = args.yolov5_dir.resolve()
    if not (repo / "train.py").is_file():
        parser.error("--yolov5-dir 必须指向包含 train.py 的 YOLOv5 源码目录")
    data_path = args.data.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    dataset_root = (args.dataset_root or data_path.parent).resolve()
    config["path"] = str(dataset_root)
    names = config["names"]
    classes = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    if classes != ["apple", "banana", "milk_box", "coke_can", "egg", "blueberry_box"]:
        parser.error("类别及顺序必须与当前板端的 6 类食材一致")
    config["nc"] = len(classes)

    output = args.output.resolve()
    if output.exists():
        parser.error("输出目录已存在，请通过 --output 指定新目录，避免覆盖训练记录")
    output.mkdir(parents=True)
    runtime_data = output / "dataset.yaml"
    runtime_data.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    weights = str(Path(args.weights).resolve()) if Path(args.weights).is_file() else args.weights
    command = [
        sys.executable, str(repo / "train.py"),
        "--data", str(runtime_data), "--weights", weights,
        "--hyp", str(args.hyp.resolve()), "--imgsz", "416",
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--workers", str(args.workers), "--seed", "0", "--optimizer", "SGD",
        "--project", str(output), "--name", "run",
    ]
    if args.device:
        command.extend(["--device", args.device])
    subprocess.run(command, cwd=repo, check=True)
    print(f"训练权重：{output / 'run/weights/best.pt'}")


if __name__ == "__main__":
    main()
