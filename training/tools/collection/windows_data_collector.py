import os
import cv2
import time
import threading
from datetime import datetime
from flask import Flask, Response, request, jsonify

# ============================================================
# Windows 本地数据采集脚本
# 项目：研电赛智能冰箱食材识别与管理系统
#
# 功能：
# 1. Windows 直连外接 USB 摄像头
# 2. 浏览器实时预览
# 3. 网页选择类别
# 4. 开始/停止采集
# 5. 设置采集间隔
# 6. 实时统计各类别数量
# 7. 保存到 E:\rk3566_fridge_data
#
# 当前类别：
# apple, banana, milk_box, coke_can, egg, blueberry_box, negative
# ============================================================


# =========================
# 基础配置
# =========================

SAVE_ROOT = r"E:\rk3566_fridge_data"

RAW_ROOT = os.path.join(SAVE_ROOT, "raw_images")
EVENT_ROOT = os.path.join(SAVE_ROOT, "event_sequences")
LOG_ROOT = os.path.join(SAVE_ROOT, "logs")

# 摄像头编号：
# Camera 0：电脑自带摄像头
# Camera 1：外接冰箱 USB 摄像头
CAMERA_INDEX = 1

# 采集参数：尽量和 RK3566 板端保持一致
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 15

# 网页预览分辨率
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

# JPEG 质量
SAVE_JPEG_QUALITY = 95
STREAM_JPEG_QUALITY = 80

# Flask 服务
HOST = "127.0.0.1"
PORT = 8080


# =========================
# 类别配置
# =========================

RAW_CLASSES = {
    "apple": "苹果",
    "banana": "香蕉",
    "milk_box": "盒装牛奶",
    "coke_can": "罐装可乐",
    "egg": "鸡蛋",
    "blueberry_box": "盒装蓝莓",
    "negative": "负样本",
}

EVENT_CLASSES = {
    "put_in": "放入事件",
    "take_out": "取出事件",
    "partial_take": "部分取出",
    "no_change_hand": "手部经过无变化",
    "no_change_occlusion": "遮挡无变化",
    "no_change_rearrange": "整理无变化",
}

ALL_CLASSES = {}
ALL_CLASSES.update({k: ("raw", v) for k, v in RAW_CLASSES.items()})
ALL_CLASSES.update({k: ("event", v) for k, v in EVENT_CLASSES.items()})


# =========================
# 全局状态
# =========================

app = Flask(__name__)

camera_lock = threading.Lock()
state_lock = threading.Lock()

latest_frame = None
camera_opened = False
camera_error = ""

running = True

collecting = False
collect_class = None
collect_interval = 2.0
last_save_time = 0.0
saved_total = 0
last_saved_path = ""


# =========================
# 工具函数
# =========================

def ensure_dirs():
    """创建所有必要目录。"""
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(RAW_ROOT, exist_ok=True)
    os.makedirs(EVENT_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)

    for cls in RAW_CLASSES:
        os.makedirs(os.path.join(RAW_ROOT, cls), exist_ok=True)

    for cls in EVENT_CLASSES:
        os.makedirs(os.path.join(EVENT_ROOT, cls), exist_ok=True)

    os.makedirs(os.path.join(SAVE_ROOT, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "dataset", "images"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "dataset", "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "dataset", "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "dataset", "labels"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "dataset", "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "dataset", "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "backup"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "test_samples"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "test_results"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "docs"), exist_ok=True)


def get_save_dir(cls):
    """根据类别返回保存目录。"""
    if cls in RAW_CLASSES:
        return os.path.join(RAW_ROOT, cls)
    if cls in EVENT_CLASSES:
        return os.path.join(EVENT_ROOT, cls)
    return None


def get_counts():
    """统计各类别 jpg/jpeg 数量。"""
    counts = {}

    for cls in RAW_CLASSES:
        d = os.path.join(RAW_ROOT, cls)
        if os.path.exists(d):
            counts[cls] = len([
                f for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".jpeg"))
            ])
        else:
            counts[cls] = 0

    for cls in EVENT_CLASSES:
        d = os.path.join(EVENT_ROOT, cls)
        if os.path.exists(d):
            counts[cls] = len([
                f for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".jpeg"))
            ])
        else:
            counts[cls] = 0

    return counts


def write_log(message):
    """写采集日志。"""
    os.makedirs(LOG_ROOT, exist_ok=True)
    log_path = os.path.join(LOG_ROOT, "collector_log.txt")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# =========================
# 摄像头线程
# =========================

def open_camera():
    """
    打开 Windows 摄像头。
    使用 DirectShow 后端。
    """
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    # 强制 MJPG，降低带宽和 CPU 压力
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None

    return cap


def camera_worker():
    global latest_frame
    global camera_opened, camera_error
    global collecting, collect_class, collect_interval
    global last_save_time, saved_total, last_saved_path

    try:
        cap = open_camera()

        if cap is None:
            camera_opened = False
            camera_error = (
                f"摄像头打开失败：CAMERA_INDEX={CAMERA_INDEX}。"
                "请检查摄像头是否被 Windows 相机、微信、QQ、浏览器等软件占用。"
            )
            write_log(camera_error)
            return

        camera_opened = True
        camera_error = ""

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])

        write_log("Camera opened.")
        write_log(f"CAMERA_INDEX={CAMERA_INDEX}")
        write_log(f"Target: MJPG {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}@{CAPTURE_FPS}fps")
        write_log(f"Actual: {actual_w}x{actual_h}@{actual_fps:.2f}fps, FOURCC={fourcc}")

        # 摄像头预热
        for _ in range(10):
            cap.read()
            time.sleep(0.02)

        while running:
            ret, frame = cap.read()
            if not ret or frame is None:
                write_log("WARN: read frame failed")
                time.sleep(0.1)
                continue

            with camera_lock:
                latest_frame = frame.copy()

            now = time.time()

            with state_lock:
                do_collect = collecting
                cls = collect_class
                interval = collect_interval

            if do_collect and cls is not None:
                if now - last_save_time >= interval:
                    save_dir = get_save_dir(cls)

                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)

                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{cls}_{ts}.jpg"
                        path = os.path.join(save_dir, filename)

                        ok = cv2.imwrite(
                            path,
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), SAVE_JPEG_QUALITY]
                        )

                        if ok:
                            with state_lock:
                                saved_total += 1
                                last_save_time = now
                                last_saved_path = path
                            write_log(f"saved: {path}")
                        else:
                            write_log(f"WARN: save failed: {path}")

        cap.release()

    except Exception as e:
        camera_opened = False
        camera_error = repr(e)
        write_log(f"ERROR: {camera_error}")


# =========================
# 视频流
# =========================

def gen_frames():
    while True:
        with camera_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        show = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

        with state_lock:
            status_text = "COLLECTING" if collecting else "IDLE"
            cls = collect_class
            interval = collect_interval

        overlay = f"{status_text} | class={cls} | interval={interval:.1f}s"

        cv2.putText(
            show,
            overlay,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        ok, buffer = cv2.imencode(
            ".jpg",
            show,
            [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# =========================
# Web 页面
# =========================

@app.route("/")
def index():
    return r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Windows 冰箱数据采集控制台</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, "Microsoft YaHei", sans-serif;
            background: #f3f6fb;
            color: #222;
        }
        .container {
            width: 1180px;
            margin: 24px auto;
        }
        h1 {
            text-align: center;
            color: #1f4e79;
            margin-bottom: 8px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 24px;
        }
        .layout {
            display: grid;
            grid-template-columns: 700px 1fr;
            gap: 20px;
        }
        .card {
            background: #fff;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            padding: 20px;
        }
        .video {
            width: 640px;
            height: 360px;
            border-radius: 10px;
            border: 1px solid #ddd;
            display: block;
            margin: 0 auto;
            background: #111;
        }
        label {
            font-weight: bold;
            display: block;
            margin-top: 14px;
            margin-bottom: 6px;
        }
        select, input {
            width: 100%;
            box-sizing: border-box;
            padding: 10px;
            font-size: 15px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .btn-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 18px;
        }
        button {
            padding: 12px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
        }
        .start {
            background: #1f8f4d;
            color: white;
        }
        .stop {
            background: #c0392b;
            color: white;
        }
        .refresh {
            background: #1f4e79;
            color: white;
            margin-top: 12px;
            width: 100%;
        }
        .status {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 12px;
            border-radius: 10px;
            margin-top: 14px;
            line-height: 1.8;
            font-size: 14px;
        }
        .warning {
            color: #a15c00;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 12px;
        }
        .error {
            color: #b00020;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
            font-size: 14px;
        }
        th, td {
            border-bottom: 1px solid #eee;
            padding: 8px;
            text-align: left;
        }
        th {
            background: #f0f4f8;
        }
        .tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            background: #e8f1ff;
            color: #1f4e79;
            font-size: 12px;
        }
        .small {
            font-size: 13px;
            color: #666;
            line-height: 1.7;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Windows 冰箱食材识别数据采集控制台</h1>
        <div class="subtitle">
            外接 USB 摄像头编号：Camera 1｜
            保存路径：E:\rk3566_fridge_data｜
            蓝莓盒用于满/中/空状态判断
        </div>

        <div class="layout">
            <div class="card">
                <h2>实时画面</h2>
                <img class="video" src="/video_feed">
                <div class="warning">
                    注意：虽然当前在 Windows 上采集，但摄像头角度、补光、背景、食材距离应尽量接近后续 RK3566 演示环境。
                </div>
                <div class="small">
                    建议采集间隔：单目标静态图 1.5–2.0 秒；负样本 1.0–1.5 秒；手部事件帧 0.5–1.0 秒。
                    蓝莓盒满/中/空统一选择 blueberry_box，靠文件名区分状态。
                </div>
            </div>

            <div class="card">
                <h2>采集控制</h2>

                <label>选择采集对象</label>
                <select id="classSelect">
                    <optgroup label="目标检测图片 raw_images">
                        <option value="apple">苹果 apple</option>
                        <option value="banana">香蕉 banana</option>
                        <option value="milk_box">盒装牛奶 milk_box</option>
                        <option value="coke_can">罐装可乐 coke_can</option>
                        <option value="egg">鸡蛋 egg</option>
                        <option value="blueberry_box">盒装蓝莓 blueberry_box</option>
                        <option value="negative">负样本 negative</option>
                    </optgroup>
                    <optgroup label="事件过程帧 event_sequences">
                        <option value="put_in">放入事件 put_in</option>
                        <option value="take_out">取出事件 take_out</option>
                        <option value="partial_take">部分取出 partial_take</option>
                        <option value="no_change_hand">手部经过无变化 no_change_hand</option>
                        <option value="no_change_occlusion">遮挡无变化 no_change_occlusion</option>
                        <option value="no_change_rearrange">整理无变化 no_change_rearrange</option>
                    </optgroup>
                </select>

                <label>采集间隔，单位：秒</label>
                <input id="intervalInput" type="number" value="2.0" min="0.3" step="0.1">

                <div class="btn-row">
                    <button class="start" onclick="startCollect()">开始采集</button>
                    <button class="stop" onclick="stopCollect()">停止采集</button>
                </div>

                <button class="refresh" onclick="refreshStatus()">刷新状态与数量</button>

                <div class="status" id="statusBox">
                    状态加载中...
                </div>

                <h3>已采集数量</h3>
                <table>
                    <thead>
                        <tr>
                            <th>类别</th>
                            <th>数量</th>
                        </tr>
                    </thead>
                    <tbody id="countTable"></tbody>
                </table>
            </div>
        </div>
    </div>

<script>
async function startCollect() {
    const cls = document.getElementById("classSelect").value;
    const interval = parseFloat(document.getElementById("intervalInput").value);

    const resp = await fetch("/api/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({class_name: cls, interval: interval})
    });

    const data = await resp.json();
    refreshStatus();

    if (!data.ok) {
        alert("启动失败：" + data.message);
    } else {
        alert(data.message);
    }
}

async function stopCollect() {
    const resp = await fetch("/api/stop", {method: "POST"});
    const data = await resp.json();
    refreshStatus();
    alert(data.message);
}

async function refreshStatus() {
    const resp = await fetch("/api/status");
    const data = await resp.json();

    let cameraInfo = "";
    if (data.camera_opened) {
        cameraInfo = `<div>摄像头状态：<span class="tag">已打开</span></div>`;
    } else {
        cameraInfo = `<div class="error">摄像头状态：未打开｜${data.camera_error || ""}</div>`;
    }

    document.getElementById("statusBox").innerHTML = `
        ${cameraInfo}
        <div>采集状态：<span class="tag">${data.collecting ? "正在采集" : "空闲"}</span></div>
        <div>当前类别：${data.collect_class || "-"}</div>
        <div>采集间隔：${data.collect_interval} 秒</div>
        <div>本次启动后保存数量：${data.saved_total}</div>
        <div>最后保存文件：${data.last_saved_path || "-"}</div>
        <div>保存根目录：E:\\rk3566_fridge_data</div>
    `;

    const tbody = document.getElementById("countTable");
    tbody.innerHTML = "";

    for (const [key, value] of Object.entries(data.counts)) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${key}</td><td>${value}</td>`;
        tbody.appendChild(tr);
    }
}

setInterval(refreshStatus, 2000);
refreshStatus();
</script>

</body>
</html>
"""


# =========================
# API
# =========================

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/start", methods=["POST"])
def api_start():
    global collecting, collect_class, collect_interval, last_save_time

    data = request.get_json(force=True)
    cls = data.get("class_name")
    interval = float(data.get("interval", 2.0))

    if cls not in ALL_CLASSES:
        return jsonify({"ok": False, "message": "无效类别"}), 400

    if interval < 0.3:
        interval = 0.3

    save_dir = get_save_dir(cls)
    if save_dir is None:
        return jsonify({"ok": False, "message": "保存目录无效"}), 400

    os.makedirs(save_dir, exist_ok=True)

    with state_lock:
        collect_class = cls
        collect_interval = interval
        collecting = True
        last_save_time = 0.0

    write_log(f"START collect class={cls}, interval={interval}s")

    return jsonify({
        "ok": True,
        "message": f"开始采集：{cls}，间隔 {interval} 秒"
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global collecting

    with state_lock:
        collecting = False

    write_log("STOP collect")

    return jsonify({
        "ok": True,
        "message": "已停止采集"
    })


@app.route("/api/status")
def api_status():
    with state_lock:
        status = {
            "camera_opened": camera_opened,
            "camera_error": camera_error,
            "collecting": collecting,
            "collect_class": collect_class,
            "collect_interval": collect_interval,
            "saved_total": saved_total,
            "last_saved_path": last_saved_path,
            "counts": get_counts(),
        }
    return jsonify(status)


# =========================
# 主入口
# =========================

if __name__ == "__main__":
    ensure_dirs()

    write_log("============================================================")
    write_log("Windows Data Collector Starting")
    write_log(f"SAVE_ROOT={SAVE_ROOT}")
    write_log(f"RAW_ROOT={RAW_ROOT}")
    write_log(f"EVENT_ROOT={EVENT_ROOT}")
    write_log(f"CAMERA_INDEX={CAMERA_INDEX}")

    t = threading.Thread(target=camera_worker, daemon=True)
    t.start()

    print()
    print("浏览器打开：")
    print(f"http://{HOST}:{PORT}")
    print()
    print("当前摄像头编号：CAMERA_INDEX = 1")
    print("当前目标类别包含：apple, banana, milk_box, coke_can, egg, blueberry_box, negative")
    print()

    app.run(host=HOST, port=PORT, threaded=True)