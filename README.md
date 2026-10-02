# Fridge Mind RKNN

基于 YOLOv5n 和 RK3566 的冰箱食材识别与库存管理系统，支持摄像头检测、稳定计数、出入库记录、库存展示、标签信息记录和语音播报，可通过云端大模型生成库存问答与食用建议。

识别类别：苹果、香蕉、盒装牛奶、罐装可乐、鸡蛋、盒装蓝莓。

## 项目结构

```text
training/
  data/          数据集配置与类别定义；图片及标注在本地准备
  tools/         数据采集、训练、ONNX 导出和 RKNN 转换
  experiments/   实验配置、训练记录及模型文件
  logs/          本地采集日志
edge/
  app/           板端主程序
  models/        部署模型及类别文件
  config/        运行配置
  assets/        语音素材
  launch/        启动脚本及桌面入口
  tools/         单图和摄像头推理工具
  tests/         应用测试
```

## 系统架构

主机侧使用 YOLOv5 训练食材检测模型，经 ONNX 导出和 RKNN-Toolkit2 转换后部署到 RK3566。训练与转换说明见 [training/README.md](training/README.md)。

数据集图片、标注、采集日志和预测输出由 `.gitignore` 排除。训练前按训练说明在本地准备 YOLO 格式数据集。

板端通过 OpenCV 获取摄像头图像，由 RKNNLite 调用 NPU 推理。检测结果经过过滤、NMS 和多帧稳定计数后更新 SQLite 库存数据库，生成出入库事件，并在界面显示和通过语音播报。

主程序为 `edge/app/camera_inventory_ui_demo.py`，采用 OpenCV 界面；语音播放与云端请求通过后台线程处理。SQLite 保存库存、事件、食材资料、提醒和标签信息。

## 板端部署

板端运行依赖 Python、NumPy、OpenCV、Pillow、与板端 runtime 匹配的 RKNNLite，以及 `aplay`、`amixer` 和用于动态播报的 `espeak-ng`。主程序使用 Python 的 `audioop` 模块，需使用提供该模块的 Python 版本（如 Python 3.10）。

默认文件布局：

| 项目文件 | 板端位置 |
| --- | --- |
| `edge/models/` 中的 RKNN 和 `classes.txt` | `/home/ztl/fridge_project/model/` |
| `edge/app/camera_inventory_ui_demo.py` | `/home/ztl/fridge_project/scripts/` |
| `edge/config/fridge_demo_config.json` | `/home/ztl/fridge_project/scripts/` |
| `edge/assets/voice_prompts/` | `/home/ztl/fridge_project/scripts/voice_prompts/` |
| `edge/launch/start_fridge_demo.sh` | `/home/ztl/fridge_project/scripts/` |

在板端启动：

```bash
cd /home/ztl/fridge_project/scripts
python3 camera_inventory_ui_demo.py --camera /dev/video18 --fullscreen --interval 8
```

可通过 `--model`、`--classes`、`--camera` 和 `--config` 指定模型、类别、摄像头及配置文件。默认优先加载 `best_fp.rknn`。

## 运行配置

`edge/config/fridge_demo_config.json` 配置图像增强、画面质量阈值、稳定计数、摄像头、语音和云端接口。

云端接口通过 `FRIDGE_LLM_API_URL`、`FRIDGE_LLM_API_KEY`、`FRIDGE_LLM_MODEL` 环境变量或本地配置设置。本地配置可命名为 `fridge_demo_config.local.json`，启动时使用 `--config` 指定；此类文件由 Git 忽略。

固定语音素材位于 `edge/assets/voice_prompts/`。库存与出入库逻辑在板端运行，云端接口用于助手回答和建议。
