# 冰箱食材模型训练与转换

本目录提供冰箱食材识别模型的训练、ONNX 导出和 RKNN 转换工具。

处理流程：YOLO 数据集 → YOLOv5n 训练 → ONNX 导出 → RK3566 RKNN 转换 → 板端应用。

## 依赖与约定

- 训练和 ONNX 导出调用外部 [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) 源码，下面的安装命令使用 v7.0。
- 在适配的 Python/PyTorch 环境中安装 YOLOv5 仓库的 `requirements.txt` 和本目录的 `requirements.txt`。PyTorch/CUDA 按训练机器配置选择，加载 `.pt` 权重时使用兼容其保存格式的 PyTorch/YOLOv5 环境。
- RKNN 转换使用 [Rockchip RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) 的官方 wheel，建议独立 Linux 转换环境，版本与板端 RKNN runtime 配套。板端的 `rknnlite` 不能代替主机转换工具。
- 类别固定为 `apple、banana、milk_box、coke_can、egg、blueberry_box`，顺序与现有应用一致。负样本对应空标注，不是第 7 类。
- 数据集默认目录为 `training/data/datasets/fridge_food_v1`，图片和 YOLO 格式标注在本地准备。仓库保留 `fridge_food.yaml` 和 `classes.txt`；训练脚本在输出目录生成数据配置，使用当前数据集的绝对路径。
- 训练和转换产物默认写入 `training/artifacts/`；`training/experiments/` 存放实验记录，`edge/models/` 存放部署模型。

以下命令均从项目根目录执行。

## 数据集结构

```text
training/data/datasets/fridge_food_v1/
  fridge_food.yaml
  classes.txt
  images/
    train/
    val/
  labels/
    train/
    val/
```

图片与标注文件同名，例如 `images/train/apple_001.jpg` 对应 `labels/train/apple_001.txt`。每行标注格式为 `class_id x_center y_center width height`，坐标归一化到 0～1；负样本使用空标注文件。

## 1. 训练

准备外部 YOLOv5 源码，例如放到项目外的 `../yolov5`：

```bash
git clone --branch v7.0 https://github.com/ultralytics/yolov5.git ../yolov5
python -m pip install -r ../yolov5/requirements.txt
python -m pip install -r training/requirements.txt
python training/tools/train.py --yolov5-dir ../yolov5 --device 0
```

脚本调用 YOLOv5 的 `train.py`，默认使用 YOLOv5n 预训练权重、416 输入、100 轮、batch 16、SGD、seed 0，以及 `training/experiments/fridge_yolov5n_416/hyp.yaml` 中的超参数。`yolov5n.pt` 不在源码目录时上游会下载；离线可用 `--weights /path/to/yolov5n.pt` 指定本地权重。CPU 可用 `--device cpu`。

默认权重位置：`training/artifacts/fridge_yolov5n_416/run/weights/best.pt`。重复训练需通过 `--output` 指定新目录。如数据搬到别处，用 `--dataset-root` 指定数据集目录。

## 2. 导出 ONNX

```bash
python training/tools/export_onnx.py --yolov5-dir ../yolov5 \
  --weights training/artifacts/fridge_yolov5n_416/run/weights/best.pt
```

生成 `training/artifacts/deploy/best.onnx`。也可以把 `--weights` 改成已有实验的 `training/experiments/fridge_yolov5n_416/weights/best.pt`。

导出调用上游 [export.py](https://github.com/ultralytics/yolov5/blob/v7.0/export.py)，固定 batch 1、416×416、opset 12，不启用动态尺寸或 NMS。板端后处理接收已解码的单个检测输出 `[1, N, 11]`，每条预测是 `x,y,w,h,obj,6个类别概率`，过滤和 NMS 在板端执行。导出的模型需保持这一输出格式。

## 3. 转换 RKNN

在装有匹配版本 RKNN-Toolkit2 的转换环境执行：

```bash
# 浮点模型，现有应用优先使用它。
python training/tools/convert_rknn.py \
  --onnx training/artifacts/deploy/best.onnx

# 可选 INT8 模型，需要现有训练图片作为量化校准输入。
python training/tools/convert_rknn.py \
  --onnx training/artifacts/deploy/best.onnx --int8
```

产物分别为 `training/artifacts/deploy/best_fp.rknn`、`best.rknn`，并写入 `classes.txt`。INT8 默认从训练集固定随机选取最多 100 张图片，按板端的 416 letterbox 和 114 填充值生成量化校准图片；可用 `--calibration-images` 指定其他代表性图片目录。

转换依据官方 [RKNN 示例接口](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/onnx/yolov5/test.py)，目标平台为 `rk3566`，均值 0、标准差 255。板端输入保持 RGB uint8，归一化由模型完成。

## 4. 接入已有板端应用

将生成的 RKNN 模型和 `classes.txt` 放入板端 `/home/ztl/fridge_project/model/`，文件名保持 `best_fp.rknn` / `best.rknn`。ONNX 是转换中间产物，板端应用不需要加载它。现有启动脚本默认优先使用浮点模型，因此部署 INT8 时可直接通过应用的 `--model` 显式选择。

部署文件对应关系：

| 项目文件 | 板端位置 |
| --- | --- |
| `training/artifacts/deploy/*.rknn`、`classes.txt` | `/home/ztl/fridge_project/model/` |
| `edge/app/camera_inventory_ui_demo.py` | `/home/ztl/fridge_project/scripts/` |
| `edge/config/fridge_demo_config.json` | `/home/ztl/fridge_project/scripts/` |
| `edge/assets/voice_prompts/` | `/home/ztl/fridge_project/scripts/voice_prompts/` |

`edge/` 提供摄像头识别、稳定计数、库存数据库、出入库记录、界面及语音功能。
