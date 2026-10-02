"""导出适配现有板端后处理的 YOLOv5 单输出 ONNX。"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolov5-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "training/artifacts/deploy/best.onnx")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    repo = args.yolov5_dir.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    if not (repo / "export.py").is_file() or not weights.is_file():
        parser.error("需要有效的 YOLOv5 源码目录和 .pt 权重")
    if output.exists():
        parser.error("输出文件已存在，请指定新的 --output")
    output.parent.mkdir(parents=True, exist_ok=True)

    # 上游将 ONNX 写到权重旁边；通过临时目录导出到指定位置。
    with tempfile.TemporaryDirectory(prefix="fridge_onnx_") as work:
        staged_weights = Path(work) / "best.pt"
        shutil.copy2(weights, staged_weights)
        subprocess.run([
            sys.executable, str(repo / "export.py"),
            "--weights", str(staged_weights), "--include", "onnx",
            "--imgsz", "416", "416", "--batch-size", "1",
            "--device", "cpu", "--opset", str(args.opset),
        ], cwd=repo, check=True)
        # 上游某些版本会捕获导出异常，需要确认文件确实生成。
        generated = staged_weights.with_suffix(".onnx")
        if not generated.is_file():
            raise RuntimeError("YOLOv5 未生成 ONNX，请查看导出日志")
        shutil.copy2(generated, output)
    print(f"ONNX：{output}")


if __name__ == "__main__":
    main()
