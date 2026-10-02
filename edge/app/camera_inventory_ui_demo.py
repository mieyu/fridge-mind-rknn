import time
import os
import re
import json
import shlex
import sqlite3
import argparse
import subprocess
import threading
import queue
import hashlib
import sys
import wave
import warnings
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, Counter

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop


NORMALIZED_AUDIO_RATE = 48000
NORMALIZED_AUDIO_CHANNELS = 2
NORMALIZED_AUDIO_WIDTH = 2


def normalize_wav_file(
    src_path,
    dst_path,
    sample_rate=NORMALIZED_AUDIO_RATE,
    channels=NORMALIZED_AUDIO_CHANNELS,
    sample_width=NORMALIZED_AUDIO_WIDTH,
):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        try:
            src_stat = src_path.stat()
            dst_stat = dst_path.stat()
            if dst_stat.st_mtime >= src_stat.st_mtime and dst_stat.st_size > 44:
                return str(dst_path)
        except OSError:
            pass

    with wave.open(str(src_path), "rb") as src:
        src_channels = src.getnchannels()
        src_width = src.getsampwidth()
        src_rate = src.getframerate()
        frames = src.readframes(src.getnframes())

    if src_width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported wav sample width: {src_width}")

    if src_width == 1:
        frames = audioop.bias(frames, 1, -128)

    if src_width != sample_width:
        frames = audioop.lin2lin(frames, src_width, sample_width)
        src_width = sample_width

    if src_channels != channels:
        if src_channels == 1 and channels == 2:
            frames = audioop.tostereo(frames, sample_width, 1.0, 1.0)
        elif src_channels == 2 and channels == 1:
            frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        else:
            raise ValueError(f"unsupported wav channel conversion: {src_channels} -> {channels}")
        src_channels = channels

    if src_rate != sample_rate:
        frames, _ = audioop.ratecv(frames, sample_width, channels, src_rate, sample_rate, None)

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    with wave.open(str(tmp_path), "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sample_width)
        dst.setframerate(sample_rate)
        dst.writeframes(frames)
    tmp_path.replace(dst_path)
    return str(dst_path)


def run_normalize_audio_cli(argv):
    if len(argv) != 4 or argv[1] != "--normalize-audio":
        return False
    normalize_wav_file(argv[2], argv[3])
    return True


if run_normalize_audio_cli(sys.argv):
    raise SystemExit(0)

import cv2
import numpy as np
from rknnlite.api import RKNNLite

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


# ============================================================
# RK3566 Smart Fridge UI Demo
# Camera + RKNN + Stable Counting + SQLite + Voice + UI Pages
# ============================================================

MODEL_FP = "/home/ztl/fridge_project/model/best_fp.rknn"
MODEL_INT8 = "/home/ztl/fridge_project/model/best.rknn"
CLASSES_PATH = "/home/ztl/fridge_project/model/classes.txt"
DB_PATH = "/home/ztl/fridge_project/inventory.db"
CONFIG_PATH = str(Path(__file__).with_name("fridge_demo_config.json"))

CAMERA_DEVICE = "/dev/video18"

CAP_WIDTH = 640
CAP_HEIGHT = 480
CAP_FPS = 30

IMG_SIZE = 416
CONF_THRES = 0.25
IOU_THRES = 0.45
INFER_INTERVAL = 5

STABLE_WINDOW = 10
STABLE_VOTES = 7
EVENT_COOLDOWN = 2.0

BLUEBERRY_FULL_RATIO = 0.35
BLUEBERRY_HALF_RATIO = 0.08
BLUEBERRY_AREA_METHOD = "binary_dark"
BLUEBERRY_BINARY_THRESHOLD = 110
BLUEBERRY_ROI_SHRINK = 0.08
BLUEBERRY_COLOR_GUARD_RATIO = 0.02
BLUEBERRY_HSV_LOWER = np.array([90, 35, 20], dtype=np.uint8)
BLUEBERRY_HSV_UPPER = np.array([165, 255, 255], dtype=np.uint8)
LOW_LIGHT_MEAN = 45.0
LOW_CONTRAST_STD = 12.0
BLUR_VAR_THRES = 18.0
MILK_COKE_MIN_MARGIN = 0.12

PREPROCESS_ENABLED = False
PREPROCESS_METHOD = "plastic_wrap"
PREPROCESS_CLAHE_CLIP_LIMIT = 1.8
PREPROCESS_CLAHE_TILE_GRID = 8
PREPROCESS_GAMMA = 0.9
PREPROCESS_SATURATION_GAIN = 1.15
PREPROCESS_CONTRAST_ALPHA = 1.03
PREPROCESS_BRIGHTNESS_BETA = 0.0
PREPROCESS_SHARPEN_AMOUNT = 0.25
PREPROCESS_GLARE_SUPPRESS = True
PREPROCESS_GLARE_V_THRESHOLD = 235
PREPROCESS_GLARE_S_THRESHOLD = 80
PREPROCESS_GLARE_BLUR_KERNEL = 7

AUDIO_DEVICE = "plughw:1,0"
ENABLE_SPEECH = True
TTS_COMMAND = ""
SPEECH_WAV_DIR = str(Path(__file__).with_name("voice_prompts"))
SPEECH_WAV_MAP = {}
TTS_CACHE_DIR = "/tmp/fridge_tts_cache"
AUDIO_CACHE_DIR = "/tmp/fridge_audio_cache"
LLM_API_URL = ""
LLM_API_KEY = ""
LLM_MODEL = ""
RECORD_PATH = ""
RECORD_FPS = 20.0
CLOUD_ANALYSIS_QUESTION = (
    "请基于当前库存、最近事件和包装标签，生成一段80字以内的中文饮食建议。"
    "重点提醒临期、营养和优先食用顺序，适合智能冰箱语音播报。"
)
SIMULATED_VOICE_QUESTION = "请结合当前库存给出饮食建议"

DEFAULT_SPEECH_PROMPT_KEYS = {
    "系统就绪": "system_ready",
    "库存已清空": "inventory_reset",
    "库存已按当前画面初始化": "inventory_initialized",
    "当前没有稳定识别结果": "no_stable_snapshot",
    "标签录入成功，已记录保质期和营养信息": "label_saved",
    "标签识别失败，请重新对准包装标签": "label_failed",
    "助手已就绪。按 A 查询库存，按 D 获取食用建议，按 R 设置提醒。": "assistant_ready",
    "当前冰箱库存为空。": "inventory_empty",
    "当前冰箱库存为空，暂时不需要食用提醒。": "inventory_empty",
    "当前库存为空，暂不设置提醒。": "inventory_empty",
}

WINDOW_NAME = "RK3566 Smart Fridge AI"
UI_W = 1000
UI_H = 600

PAGE_DETECT = "detect"
PAGE_INVENTORY = "inventory"
PAGE_EVENTS = "events"
PAGE_ASSISTANT = "assistant"
PAGE_LABEL = "label"

CLASS_ORDER = [
    "apple",
    "banana",
    "milk_box",
    "coke_can",
    "egg",
    "blueberry_box",
]

CLASS_SPEAK_NAME = {
    "apple": "苹果",
    "banana": "香蕉",
    "milk_box": "盒装牛奶",
    "coke_can": "罐装饮料",
    "egg": "鸡蛋",
    "blueberry_box": "蓝莓",
}

CLASS_SPEECH_EVENT_KEY = {
    "apple": "apple",
    "banana": "banana",
    "milk_box": "milk",
    "coke_can": "coke",
    "egg": "egg",
    "blueberry_box": "blueberry",
}

CLASS_DISPLAY_NAME = {
    "apple": "苹果",
    "banana": "香蕉",
    "milk_box": "盒装牛奶",
    "coke_can": "罐装饮料",
    "egg": "鸡蛋",
    "blueberry_box": "蓝莓盒",
}

CLASS_UNIT = {
    "apple": "个",
    "banana": "根",
    "milk_box": "盒",
    "coke_can": "罐",
    "egg": "个",
    "blueberry_box": "盒",
}

SHELF_LIFE_DAYS = {
    "apple": 7,
    "banana": 3,
    "milk_box": 3,
    "coke_can": 30,
    "egg": 14,
    "blueberry_box": 5,
}

STORAGE_TIP = {
    "apple": "冷藏保存，避免与气味较重食材混放。",
    "banana": "建议尽快食用，冷藏可能导致表皮变黑。",
    "milk_box": "开封后优先食用，注意查看包装日期。",
    "coke_can": "密封饮品可较长时间保存，避免冷冻。",
    "egg": "冷藏保存，食用前检查外壳完整性。",
    "blueberry_box": "保持干燥冷藏，软化或渗水时优先处理。",
}

DEFAULT_NUTRITION_INFO = {
    "apple": {
        "summary": "苹果含膳食纤维、果胶和维生素C，适合作为日常水果。",
        "nutrition": [
            "能量：约52 kcal / 100g",
            "碳水：约14 g / 100g",
            "膳食纤维：约2.4 g / 100g",
            "维生素C：约4.6 mg / 100g",
        ],
        "advice": "建议冷藏保存，7天内食用；表皮软化或碰伤时优先处理。",
    },
    "banana": {
        "summary": "香蕉富含碳水和钾，适合快速补充能量。",
        "nutrition": [
            "能量：约89 kcal / 100g",
            "碳水：约23 g / 100g",
            "膳食纤维：约2.6 g / 100g",
            "钾：约358 mg / 100g",
        ],
        "advice": "建议3天内食用；表皮变黑不一定变质，但软烂渗液时不要食用。",
    },
    "egg": {
        "summary": "鸡蛋含优质蛋白和脂类，是常见高蛋白食材。",
        "nutrition": [
            "能量：约143 kcal / 100g",
            "蛋白质：约13 g / 100g",
            "脂肪：约10 g / 100g",
            "胆碱：含量较高",
        ],
        "advice": "建议冷藏保存，14天内食用；破壳或异味时不要食用。",
    },
    "blueberry_box": {
        "summary": "蓝莓含花青素、维生素C和膳食纤维。",
        "nutrition": [
            "能量：约57 kcal / 100g",
            "碳水：约14.5 g / 100g",
            "膳食纤维：约2.4 g / 100g",
            "维生素C：约9.7 mg / 100g",
        ],
        "advice": "建议保持干燥冷藏，5天内食用；软化或渗水时优先处理。",
    },
}

CLASS_COLOR = {
    "apple": (60, 90, 255),
    "banana": (0, 220, 255),
    "milk_box": (255, 210, 120),
    "coke_can": (60, 60, 255),
    "egg": (230, 230, 210),
    "blueberry_box": (255, 120, 180),
}

UI_STATE = {
    "page": PAGE_DETECT,
    "buttons": [],
    "pending_action": None,
    "label_target": "milk_box",
    "label_capture_path": "",
    "label_record": None,
    "label_status": "请将包装标签对准取景框。",
    "inventory_selected_class": "apple",
    "inventory_detail_scroll": 0,
}

LABEL_SUPPORTED_CLASSES = ["milk_box", "coke_can"]
LABEL_SUCCESS_TEXT = "标签录入成功，已记录保质期和营养信息"
LABEL_FAILED_TEXT = "标签识别失败，请重新对准包装标签"
LABEL_CAPTURE_DIR = str(Path(__file__).with_name("label_captures"))
LABEL_SAMPLE_OCR_TEXT = {
    "milk_box": """品牌：演示牧场
产品名称：纯牛奶
保质期：6个月
生产日期：2026-06-01
到期日期：2026-12-01
营养成分表 每100mL
能量：280 kJ
蛋白质：3.2 g
脂肪：3.6 g
碳水化合物：4.8 g
钠：60 mg
""",
    "coke_can": """品牌：演示汽水
产品名称：罐装可乐
保质期：12个月
生产日期：2026-05-20
到期日期：2027-05-20
营养成分表 每100mL
能量：180 kJ
蛋白质：0 g
脂肪：0 g
碳水化合物：10.6 g
钠：12 mg
""",
}


# ============================================================
# Runtime config
# ============================================================

def load_runtime_config(path):
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        print("[INFO] config not found, using defaults:", config_path)
        return {}
    config_path = config_path.resolve()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid config JSON: {config_path}, {e}") from e

    if not isinstance(config, dict):
        raise RuntimeError(f"config root must be an object: {config_path}")

    config["_config_dir"] = str(config_path.parent)
    print("[INFO] config:", config_path)
    return config


def resolve_runtime_path(value, base_dir):
    path_text = str(value).strip()
    if not path_text:
        return path_text

    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str(Path(base_dir) / path)


def config_section(config, name):
    section = config.get(name, {})
    return section if isinstance(section, dict) else {}


def config_value(section, key, default):
    value = section.get(key, default)
    return default if value is None else value


def config_array(section, key, default):
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, list) or len(value) != 3:
        print(f"[WARN] ignore invalid config array: {key}={value}")
        return default
    return np.array([int(v) for v in value], dtype=np.uint8)


def apply_runtime_config(config):
    global BLUEBERRY_FULL_RATIO, BLUEBERRY_HALF_RATIO, BLUEBERRY_AREA_METHOD
    global BLUEBERRY_BINARY_THRESHOLD, BLUEBERRY_ROI_SHRINK
    global BLUEBERRY_COLOR_GUARD_RATIO, BLUEBERRY_HSV_LOWER, BLUEBERRY_HSV_UPPER
    global LOW_LIGHT_MEAN, LOW_CONTRAST_STD, BLUR_VAR_THRES
    global PREPROCESS_ENABLED, PREPROCESS_METHOD, PREPROCESS_CLAHE_CLIP_LIMIT
    global PREPROCESS_CLAHE_TILE_GRID, PREPROCESS_GAMMA, PREPROCESS_SATURATION_GAIN
    global PREPROCESS_CONTRAST_ALPHA, PREPROCESS_BRIGHTNESS_BETA, PREPROCESS_SHARPEN_AMOUNT
    global PREPROCESS_GLARE_SUPPRESS, PREPROCESS_GLARE_V_THRESHOLD
    global PREPROCESS_GLARE_S_THRESHOLD, PREPROCESS_GLARE_BLUR_KERNEL
    global STABLE_WINDOW, STABLE_VOTES, EVENT_COOLDOWN, INFER_INTERVAL, CONF_THRES
    global MILK_COKE_MIN_MARGIN
    global CAMERA_DEVICE, CAP_WIDTH, CAP_HEIGHT, CAP_FPS
    global AUDIO_DEVICE, ENABLE_SPEECH, TTS_COMMAND, SPEECH_WAV_DIR, SPEECH_WAV_MAP
    global LLM_API_URL, LLM_API_KEY, LLM_MODEL, RECORD_PATH, RECORD_FPS

    blueberry = config_section(config, "blueberry")
    BLUEBERRY_FULL_RATIO = float(config_value(blueberry, "full_ratio", BLUEBERRY_FULL_RATIO))
    BLUEBERRY_HALF_RATIO = float(config_value(blueberry, "half_ratio", BLUEBERRY_HALF_RATIO))
    BLUEBERRY_AREA_METHOD = str(config_value(blueberry, "method", BLUEBERRY_AREA_METHOD)).strip()
    BLUEBERRY_BINARY_THRESHOLD = int(config_value(blueberry, "binary_threshold", BLUEBERRY_BINARY_THRESHOLD))
    BLUEBERRY_ROI_SHRINK = float(config_value(blueberry, "roi_shrink", BLUEBERRY_ROI_SHRINK))
    BLUEBERRY_COLOR_GUARD_RATIO = float(config_value(blueberry, "color_guard_ratio", BLUEBERRY_COLOR_GUARD_RATIO))
    BLUEBERRY_HSV_LOWER = config_array(blueberry, "hsv_lower", BLUEBERRY_HSV_LOWER)
    BLUEBERRY_HSV_UPPER = config_array(blueberry, "hsv_upper", BLUEBERRY_HSV_UPPER)

    quality = config_section(config, "quality")
    LOW_LIGHT_MEAN = float(config_value(quality, "low_light_mean", LOW_LIGHT_MEAN))
    LOW_CONTRAST_STD = float(config_value(quality, "low_contrast_std", LOW_CONTRAST_STD))
    BLUR_VAR_THRES = float(config_value(quality, "blur_var_threshold", BLUR_VAR_THRES))

    preprocess = config_section(config, "preprocess")
    PREPROCESS_ENABLED = bool(config_value(preprocess, "enabled", PREPROCESS_ENABLED))
    PREPROCESS_METHOD = str(config_value(preprocess, "method", PREPROCESS_METHOD)).strip()
    PREPROCESS_CLAHE_CLIP_LIMIT = float(config_value(preprocess, "clahe_clip_limit", PREPROCESS_CLAHE_CLIP_LIMIT))
    PREPROCESS_CLAHE_TILE_GRID = int(config_value(preprocess, "clahe_tile_grid", PREPROCESS_CLAHE_TILE_GRID))
    PREPROCESS_GAMMA = float(config_value(preprocess, "gamma", PREPROCESS_GAMMA))
    PREPROCESS_SATURATION_GAIN = float(config_value(preprocess, "saturation_gain", PREPROCESS_SATURATION_GAIN))
    PREPROCESS_CONTRAST_ALPHA = float(config_value(preprocess, "contrast_alpha", PREPROCESS_CONTRAST_ALPHA))
    PREPROCESS_BRIGHTNESS_BETA = float(config_value(preprocess, "brightness_beta", PREPROCESS_BRIGHTNESS_BETA))
    PREPROCESS_SHARPEN_AMOUNT = float(config_value(preprocess, "sharpen_amount", PREPROCESS_SHARPEN_AMOUNT))
    PREPROCESS_GLARE_SUPPRESS = bool(config_value(preprocess, "glare_suppress", PREPROCESS_GLARE_SUPPRESS))
    PREPROCESS_GLARE_V_THRESHOLD = int(config_value(preprocess, "glare_v_threshold", PREPROCESS_GLARE_V_THRESHOLD))
    PREPROCESS_GLARE_S_THRESHOLD = int(config_value(preprocess, "glare_s_threshold", PREPROCESS_GLARE_S_THRESHOLD))
    PREPROCESS_GLARE_BLUR_KERNEL = int(config_value(preprocess, "glare_blur_kernel", PREPROCESS_GLARE_BLUR_KERNEL))

    runtime = config_section(config, "runtime")
    STABLE_WINDOW = int(config_value(runtime, "stable_window", STABLE_WINDOW))
    STABLE_VOTES = int(config_value(runtime, "stable_votes", STABLE_VOTES))
    EVENT_COOLDOWN = float(config_value(runtime, "event_cooldown", EVENT_COOLDOWN))
    INFER_INTERVAL = int(config_value(runtime, "infer_interval", INFER_INTERVAL))
    CONF_THRES = float(config_value(runtime, "conf_threshold", CONF_THRES))
    MILK_COKE_MIN_MARGIN = float(config_value(runtime, "milk_coke_min_margin", MILK_COKE_MIN_MARGIN))

    camera = config_section(config, "camera")
    CAMERA_DEVICE = str(config_value(camera, "device", CAMERA_DEVICE))
    CAP_WIDTH = int(config_value(camera, "width", CAP_WIDTH))
    CAP_HEIGHT = int(config_value(camera, "height", CAP_HEIGHT))
    CAP_FPS = int(config_value(camera, "fps", CAP_FPS))

    speech = config_section(config, "speech")
    ENABLE_SPEECH = bool(config_value(speech, "enabled", ENABLE_SPEECH))
    AUDIO_DEVICE = str(config_value(speech, "audio_device", AUDIO_DEVICE))
    TTS_COMMAND = str(config_value(speech, "tts_command", TTS_COMMAND))
    config_dir = config.get("_config_dir", Path(__file__).parent)
    SPEECH_WAV_DIR = resolve_runtime_path(config_value(speech, "wav_dir", SPEECH_WAV_DIR), config_dir)
    wav_map = speech.get("wav_map", SPEECH_WAV_MAP)
    SPEECH_WAV_MAP = wav_map if isinstance(wav_map, dict) else SPEECH_WAV_MAP

    llm = config_section(config, "llm")
    LLM_API_URL = str(config_value(llm, "api_url", LLM_API_URL))
    LLM_API_KEY = str(config_value(llm, "api_key", LLM_API_KEY))
    LLM_MODEL = str(config_value(llm, "model", LLM_MODEL))

    record = config_section(config, "record")
    RECORD_PATH = str(config_value(record, "path", RECORD_PATH))
    RECORD_FPS = float(config_value(record, "fps", RECORD_FPS))


# ============================================================
# Basic helpers
# ============================================================

def choose_default_model():
    if Path(MODEL_FP).exists():
        return MODEL_FP
    return MODEL_INT8


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def time_str():
    return time.strftime("%H:%M:%S")


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def normalize_count_value(value):
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return int(rounded)
    return round(value, 1)


def format_count_value(value):
    value = normalize_count_value(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.1f}".rstrip("0").rstrip(".")


def format_signed_count_value(value):
    value = normalize_count_value(value)
    text = format_count_value(abs(value))
    return f"+{text}" if value > 0 else f"-{text}"


def empty_counts():
    return {name: 0 for name in CLASS_ORDER}


def counts_to_signature(counts):
    return tuple(normalize_count_value(counts.get(name, 0)) for name in CLASS_ORDER)


def signature_to_counts(sig):
    return {name: normalize_count_value(sig[i]) for i, name in enumerate(CLASS_ORDER)}


def num_word(n):
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(int(n), str(int(n)))


def food_display_name(class_name):
    return CLASS_DISPLAY_NAME.get(class_name, class_name)


def food_unit(class_name):
    return CLASS_UNIT.get(class_name, "件")


def food_shelf_life_days(class_name):
    return int(SHELF_LIFE_DAYS.get(class_name, 7))


def format_food_amount(class_name, count):
    display = "蓝莓" if class_name == "blueberry_box" else food_display_name(class_name)
    return f"{display}{format_food_count(class_name, count)}"


def format_food_count(class_name, count):
    count = normalize_count_value(count)
    if class_name == "blueberry_box" and abs(float(count) - 0.5) < 1e-6:
        return "半盒"
    return f"{format_count_value(count)}{food_unit(class_name)}"


def build_speak_text(class_name, delta, new_count):
    name = CLASS_SPEAK_NAME.get(class_name, class_name)
    if delta > 0:
        return f"{name}增加{format_food_count(class_name, delta)}，当前{format_food_count(class_name, new_count)}"
    return f"{name}减少{format_food_count(class_name, abs(delta))}，当前{format_food_count(class_name, new_count)}"


def speech_event_prompt_key(text):
    text = str(text).strip()
    for class_name, speak_name in CLASS_SPEAK_NAME.items():
        if not text.startswith(speak_name):
            continue

        prefix = CLASS_SPEECH_EVENT_KEY.get(class_name, class_name)
        if class_name == "blueberry_box" and "半盒" in text:
            return "blueberry_half"
        if "增加" in text:
            return f"{prefix}_add"
        if "减少" in text:
            return f"{prefix}_remove"
        return None
    return None


def speech_special_prompt_key(text):
    text = str(text).strip()
    if text.startswith("已设置") and "食用提醒" in text:
        return "reminder_saved"
    return None


def extract_label_line_value(text, label):
    pattern = rf"{re.escape(label)}\s*[:：]\s*([^\n\r]+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def extract_label_metric(text, label, unit_pattern):
    pattern = rf"{re.escape(label)}\s*[:：]\s*([\d.]+)\s*({unit_pattern})"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1).strip()
    unit = match.group(2).strip()
    return f"{value} {unit}"


def build_label_advice(class_name, record):
    if class_name == "milk_box":
        protein = record.get("protein", "")
        shelf = record.get("shelf_life_text", "")
        return f"牛奶含蛋白质{protein}，保质期{shelf}，建议冷藏保存并优先饮用。"
    if class_name == "coke_can":
        carbohydrate = record.get("carbohydrate", "")
        calories = record.get("calories", "")
        return f"饮料能量{calories}，碳水化合物{carbohydrate}，建议适量饮用。"
    return "建议按包装标签信息合理安排食用。"


def parse_label_ocr_text(class_name, ocr_text):
    class_name = str(class_name).strip()
    ocr_text = str(ocr_text or "").strip()
    if class_name not in LABEL_SUPPORTED_CLASSES or not ocr_text:
        return None

    record = {
        "class_name": class_name,
        "product_name": extract_label_line_value(ocr_text, "产品名称"),
        "brand": extract_label_line_value(ocr_text, "品牌"),
        "shelf_life_text": extract_label_line_value(ocr_text, "保质期"),
        "production_date": extract_label_line_value(ocr_text, "生产日期"),
        "expiry_date": extract_label_line_value(ocr_text, "到期日期"),
        "nutrition_text": "",
        "calories": extract_label_metric(ocr_text, "能量", r"kJ|千焦"),
        "protein": extract_label_metric(ocr_text, "蛋白质", r"g|克"),
        "fat": extract_label_metric(ocr_text, "脂肪", r"g|克"),
        "carbohydrate": extract_label_metric(ocr_text, "碳水化合物", r"g|克"),
        "sodium": extract_label_metric(ocr_text, "钠", r"mg|毫克"),
        "ocr_text": ocr_text,
        "image_path": "",
    }
    nutrition_lines = []
    for label in ["能量", "蛋白质", "脂肪", "碳水化合物", "钠"]:
        value = extract_label_line_value(ocr_text, label)
        if value:
            nutrition_lines.append(f"{label}：{value}")
    record["nutrition_text"] = "\n".join(nutrition_lines)
    record["advice"] = build_label_advice(class_name, record)

    required = ["product_name", "shelf_life_text", "calories"]
    if any(not record.get(key) for key in required):
        return None
    return record


def build_label_summary(record):
    if not record:
        return ""
    parts = []
    product_name = record.get("product_name", "")
    shelf_life = record.get("shelf_life_text", "")
    protein = record.get("protein", "")
    carbohydrate = record.get("carbohydrate", "")
    if product_name:
        parts.append(product_name)
    if shelf_life:
        parts.append(f"保质期{shelf_life}")
    if protein:
        parts.append(f"蛋白质{protein}")
    if carbohydrate:
        parts.append(f"碳水{carbohydrate}")
    return "，".join(parts)


def inventory_item_by_class(db, class_name):
    for item in db.get_inventory_items():
        if item["class_name"] == class_name:
            return item
    return None


def build_inventory_detail(db, class_name):
    if class_name not in CLASS_ORDER:
        class_name = CLASS_ORDER[0]

    item = inventory_item_by_class(db, class_name) or {
        "class_name": class_name,
        "display_name": food_display_name(class_name),
        "count": 0,
        "unit": food_unit(class_name),
        "latest_label": None,
        "shelf_life_days": food_shelf_life_days(class_name),
        "storage_tip": STORAGE_TIP.get(class_name, ""),
    }
    display_name = item["display_name"]
    count = normalize_count_value(item.get("count", 0))
    lines = [
        f"当前库存：{format_food_count(class_name, count)}",
        f"默认建议：{item.get('shelf_life_days', food_shelf_life_days(class_name))}天内食用",
    ]

    if count <= 0:
        return {
            "class_name": class_name,
            "title": display_name,
            "status": "no_stock",
            "metadata": [("库存状态", "未入库")],
            "nutrition_rows": [],
            "lines": [
                "当前没有库存。",
                "入库后可在此查看保质期、营养信息和食用建议。",
            ],
        }

    latest_label = item.get("latest_label")
    if class_name in LABEL_SUPPORTED_CLASSES:
        if not latest_label:
            return {
                "class_name": class_name,
                "title": display_name,
                "status": "missing_label",
                "metadata": [
                    ("当前库存", format_food_count(class_name, count)),
                    ("标签状态", "尚未完成标签录入"),
                ],
                "nutrition_rows": [],
                "lines": lines + [
                    "已入库，但尚未完成标签录入。",
                    "请进入“标签”页面抓拍包装标签并保存。",
                    "完成后这里会显示保质期、生产日期和营养成分。",
                ],
            }

        metadata = [
            ("当前库存", format_food_count(class_name, count)),
            ("产品", latest_label.get("product_name", "-")),
            ("品牌", latest_label.get("brand", "-")),
            ("保质期", latest_label.get("shelf_life_text", "-")),
            ("生产日期", latest_label.get("production_date", "-")),
            ("到期日期", latest_label.get("expiry_date", "-")),
        ]
        nutrition_rows = [
            ("能量", latest_label.get("calories", "-")),
            ("蛋白质", latest_label.get("protein", "-")),
            ("脂肪", latest_label.get("fat", "-")),
            ("碳水", latest_label.get("carbohydrate", "-")),
            ("钠", latest_label.get("sodium", "-")),
        ]
        label_lines = [f"{key}：{value}" for key, value in metadata[1:]]
        label_lines.extend(f"{key}：{value}" for key, value in nutrition_rows)
        label_lines.append(f"建议：{latest_label.get('advice', '')}")
        return {
            "class_name": class_name,
            "title": display_name,
            "status": "ok",
            "metadata": metadata,
            "nutrition_rows": nutrition_rows,
            "lines": lines + label_lines,
        }

    nutrition = DEFAULT_NUTRITION_INFO.get(class_name, {})
    nutrition_rows = []
    for text in nutrition.get("nutrition", []):
        if "：" in text:
            key, value = text.split("：", 1)
            nutrition_rows.append((key.strip(), value.strip()))
        else:
            nutrition_rows.append(("营养", text))
    metadata = [
        ("当前库存", format_food_count(class_name, count)),
        ("来源", "本地默认营养库"),
        ("标签要求", "无需标签录入"),
        ("建议天数", f"{item.get('shelf_life_days', food_shelf_life_days(class_name))}天内食用"),
    ]
    return {
        "class_name": class_name,
        "title": display_name,
        "status": "ok",
        "metadata": metadata,
        "nutrition_rows": nutrition_rows,
        "lines": lines + [
            "营养来源：本地默认营养库，无需标签录入。",
            nutrition.get("summary", ""),
        ] + nutrition.get("nutrition", []) + [
            f"建议：{nutrition.get('advice') or item.get('storage_tip', '')}",
        ],
    }


def build_inventory_answer(db):
    items = [item for item in db.get_inventory_items() if item["count"] > 0]
    if not items:
        return "当前冰箱库存为空。"

    amount_text = "、".join(format_food_amount(item["class_name"], item["count"]) for item in items)
    advice_parts = []
    for item in items:
        label_summary = build_label_summary(item.get("latest_label"))
        if label_summary:
            advice_parts.append(f"{item['display_name']}最近标签：{label_summary}")
        else:
            advice_parts.append(f"建议{item['display_name']}{item['shelf_life_days']}天内食用")
    advice_text = "；".join(advice_parts)
    return f"当前冰箱库存有：{amount_text}。{advice_text}。"


def build_advice_answer(db):
    items = [item for item in db.get_inventory_items() if item["count"] > 0]
    if not items:
        return "当前冰箱库存为空，暂时不需要食用提醒。"

    lines = []
    for item in items:
        latest_label = item.get("latest_label")
        if latest_label:
            lines.append(
                f"{item['display_name']}{format_food_count(item['class_name'], item['count'])}，"
                f"{build_label_summary(latest_label)}，{latest_label.get('advice', '')}"
            )
        else:
            lines.append(
                f"{item['display_name']}{format_food_count(item['class_name'], item['count'])}，"
                f"建议{item['shelf_life_days']}天内食用，{item['storage_tip']}"
            )
    return "；".join(lines) + "。"


def build_reminder_answer(db):
    count = db.set_default_reminders()
    if count <= 0:
        return "当前库存为空，暂不设置提醒。"
    return f"已设置{count}条食用提醒，请按建议时间优先处理临期食材。"


def save_label_capture_record(db, speaker, class_name, image_path):
    if not str(image_path or "").strip():
        speaker.say(LABEL_FAILED_TEXT, prompt_key="label_failed")
        return False, None

    ocr_text = LABEL_SAMPLE_OCR_TEXT.get(class_name, "")
    record = parse_label_ocr_text(class_name, ocr_text)
    if not record:
        speaker.say(LABEL_FAILED_TEXT, prompt_key="label_failed")
        return False, None

    record["image_path"] = str(image_path or "")
    record_id = db.save_label_record(record)
    record["id"] = record_id
    record["created_at"] = now_str()
    speaker.say(LABEL_SUCCESS_TEXT, prompt_key="label_saved")
    return True, record


def save_label_frame(frame, capture_dir=LABEL_CAPTURE_DIR):
    if frame is None:
        return ""
    Path(capture_dir).mkdir(parents=True, exist_ok=True)
    filename = f"label_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_path = str(Path(capture_dir) / filename)
    ok = cv2.imwrite(output_path, frame)
    return output_path if ok else ""


def clear_label_ui_state(class_name=None, status="标签信息已清空。"):
    target = UI_STATE.get("label_target", "milk_box")
    if class_name is not None and class_name != target:
        return
    UI_STATE["label_capture_path"] = ""
    UI_STATE["label_record"] = None
    UI_STATE["label_status"] = status


# ============================================================
# Voice speaker
# ============================================================

class Speaker:
    def __init__(
        self,
        enable=True,
        audio_device="plughw:1,0",
        tts_command="",
        wav_dir="",
        wav_map=None,
        command_retries=2,
        retry_delay=0.15,
        command_timeout=8.0,
        tts_cache_dir=TTS_CACHE_DIR,
        audio_cache_dir=AUDIO_CACHE_DIR,
    ):
        self.enable = enable
        self.audio_device = audio_device
        self.tts_command = tts_command.strip()
        self.wav_dir = str(wav_dir).strip()
        self.wav_map = dict(wav_map or {})
        self.command_retries = max(1, int(command_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.command_timeout = max(0.01, float(command_timeout))
        self.tts_cache_dir = str(tts_cache_dir).strip() or TTS_CACHE_DIR
        self.audio_cache_dir = str(audio_cache_dir).strip() or AUDIO_CACHE_DIR
        self.q = queue.Queue()
        if self.enable:
            self.init_audio_device()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def say(self, text, prompt_key=None):
        print("[SPEAK]", text)
        if self.enable:
            self.q.put((text, prompt_key))
        else:
            print("[SPEECH] disabled, skip audio output")

    def build_audio_init_commands(self):
        match = re.match(r"^(?:plug)?hw:(\d+),", str(self.audio_device).strip())
        if not match:
            return []
        card = match.group(1)
        if card == "1":
            return [f"amixer -c {card} sset 'ELD Bypass' on"]
        return []

    def init_audio_device(self):
        for cmd in self.build_audio_init_commands():
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3,
                )
            except Exception as e:
                print("[WARN] audio init command failed:", e)
                continue
            if getattr(result, "returncode", 1) != 0:
                stderr = (getattr(result, "stderr", "") or "").strip().replace("\n", " ")[:240]
                print(f"[WARN] audio init failed (rc={result.returncode}): {stderr}")
            else:
                print("[INFO] audio init:", cmd)

    def prompt_key_for_text(self, text, prompt_key=None):
        text = str(text).strip()
        prompt_key = str(prompt_key or "").strip()
        if prompt_key and prompt_key in self.wav_map:
            return prompt_key

        if text in self.wav_map:
            return text

        prompt_key = DEFAULT_SPEECH_PROMPT_KEYS.get(text)
        if prompt_key and prompt_key in self.wav_map:
            return prompt_key

        special_key = speech_special_prompt_key(text)
        if special_key and special_key in self.wav_map:
            return special_key

        event_key = speech_event_prompt_key(text)
        if event_key and event_key in self.wav_map:
            return event_key

        return None

    def wav_path_for_text(self, text, prompt_key=None):
        if not self.wav_dir:
            return None

        prompt_key = self.prompt_key_for_text(text, prompt_key)
        if prompt_key is None:
            return None

        wav_name = str(self.wav_map.get(prompt_key, "")).strip()
        if not wav_name:
            return None

        wav_path = Path(wav_name)
        if not wav_path.is_absolute():
            wav_path = Path(self.wav_dir) / wav_name

        if wav_path.exists():
            return str(wav_path)
        print(f"[WARN] wav prompt missing: {prompt_key} -> {wav_path}")
        return None

    def tts_cache_path(self, text):
        digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]
        return str(Path(self.tts_cache_dir) / f"tts_{digest}.wav")

    def normalized_audio_cache_path(self, source_path, cache_key=None):
        source_path = str(source_path)
        if cache_key is None:
            try:
                stat = os.stat(source_path)
                cache_key = f"{source_path}:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                cache_key = source_path
        digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()[:16]
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_path).stem).strip("._") or "audio"
        return str(Path(self.audio_cache_dir) / f"{stem}_{digest}_48k2ch.wav")

    def build_normalize_audio_command(self, source_path, normalized_path):
        script_path = shlex.quote(str(Path(__file__).resolve()))
        return (
            f"python3 {script_path} --normalize-audio "
            f"{shlex.quote(str(source_path))} {shlex.quote(str(normalized_path))}"
        )

    def build_wav_play_command(self, wav_path, cache_key=None):
        normalized_path = self.normalized_audio_cache_path(wav_path, cache_key=cache_key)
        return (
            f"{self.build_normalize_audio_command(wav_path, normalized_path)} && "
            f"aplay -D {shlex.quote(self.audio_device)} {shlex.quote(normalized_path)}"
        )

    def build_tts_command(self, text):
        quoted = shlex.quote(str(text))
        if self.tts_command:
            cmd = self.tts_command.replace("{text}", quoted)
            cmd = cmd.replace("{audio_device}", shlex.quote(self.audio_device))
            if "{text}" not in self.tts_command:
                cmd = f"{self.tts_command} {quoted}"
            return cmd

        cache_dir = shlex.quote(self.tts_cache_dir)
        audio_cache_dir = shlex.quote(self.audio_cache_dir)
        raw_wav_path = self.tts_cache_path(text)
        normalized_path = self.normalized_audio_cache_path(raw_wav_path, cache_key=f"tts:{str(text)}")
        raw_wav = shlex.quote(raw_wav_path)
        device = shlex.quote(self.audio_device)
        return (
            f"mkdir -p {cache_dir} {audio_cache_dir} && "
            f"([ -f {raw_wav} ] || espeak-ng -v zh -w {raw_wav} {quoted}) && "
            f"{self.build_normalize_audio_command(raw_wav_path, normalized_path)} && "
            f"aplay -D {device} {shlex.quote(normalized_path)}"
        )

    def build_speech_commands(self, text, prompt_key=None):
        commands = []
        wav_path = self.wav_path_for_text(text, prompt_key)
        if wav_path:
            commands.append(("wav", self.build_wav_play_command(wav_path)))
        commands.append(("tts", self.build_tts_command(text)))
        return commands

    def build_speech_command(self, text, prompt_key=None):
        return self.build_speech_commands(text, prompt_key)[0][1]

    def play_text(self, text, prompt_key=None):
        for kind, cmd in self.build_speech_commands(text, prompt_key):
            attempts = self.command_retries if kind == "wav" else 1
            for attempt in range(1, attempts + 1):
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.command_timeout,
                    )
                except subprocess.TimeoutExpired:
                    print(
                        f"[WARN] speech command timeout ({kind}, try={attempt}/{attempts}, "
                        f"timeout={self.command_timeout}s)"
                    )
                    result = None
                except Exception as e:
                    print(f"[WARN] speech command failed ({kind}, try={attempt}/{attempts}):", e)
                    result = None

                if result is not None and getattr(result, "returncode", 1) == 0:
                    return True

                if result is not None:
                    stderr = (getattr(result, "stderr", "") or "").strip()
                    if stderr:
                        stderr = stderr.replace("\n", " ")[:240]
                        print(
                            f"[WARN] speech command failed ({kind}, try={attempt}/{attempts}, "
                            f"rc={result.returncode}): {stderr}"
                        )
                    else:
                        print(f"[WARN] speech command failed ({kind}, try={attempt}/{attempts}, rc={result.returncode})")

                if attempt < attempts and self.retry_delay > 0:
                    time.sleep(self.retry_delay)
        return False

    def _worker(self):
        while True:
            item = self.q.get()
            if isinstance(item, tuple):
                text, prompt_key = item
            else:
                text, prompt_key = item, None
            try:
                self.play_text(text, prompt_key)
            except Exception as e:
                print("[WARN] speech failed:", e)
            finally:
                self.q.task_done()


class UIRecorder:
    def __init__(self, output_path="", fps=20):
        self.output_path = str(output_path).strip()
        self.fps = float(fps)
        self.writer = None
        self.frame_size = None

    def enabled(self):
        return bool(self.output_path)

    def _fourcc(self):
        suffix = Path(self.output_path).suffix.lower()
        if suffix == ".mp4":
            return cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter_fourcc(*"MJPG")

    def write(self, frame):
        if not self.enabled():
            return

        h, w = frame.shape[:2]
        size = (int(w), int(h))
        if self.writer is None:
            self.frame_size = size
            self.writer = cv2.VideoWriter(self.output_path, self._fourcc(), self.fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to open recorder: {self.output_path}")

        if size != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)

        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


# ============================================================
# SQLite
# ============================================================

class InventoryDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.RLock()
        self._init_tables()

    def _init_tables(self):
        with self.lock:
            cur = self.conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                class_name TEXT PRIMARY KEY,
                count REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                class_name TEXT NOT NULL,
                delta REAL NOT NULL,
                old_count REAL NOT NULL,
                new_count REAL NOT NULL,
                event_type TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS food_profiles (
                class_name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                unit TEXT NOT NULL,
                shelf_life_days INTEGER NOT NULL,
                storage_tip TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                remind_at TEXT NOT NULL,
                message TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS label_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                product_name TEXT,
                brand TEXT,
                shelf_life_text TEXT,
                production_date TEXT,
                expiry_date TEXT,
                nutrition_text TEXT,
                calories TEXT,
                protein TEXT,
                fat TEXT,
                carbohydrate TEXT,
                sodium TEXT,
                ocr_text TEXT,
                image_path TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """)
            cur.execute("PRAGMA table_info(label_records)")
            label_columns = {row[1] for row in cur.fetchall()}
            if "active" not in label_columns:
                cur.execute("ALTER TABLE label_records ADD COLUMN active INTEGER NOT NULL DEFAULT 1")

            for name in CLASS_ORDER:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO inventory(class_name, count, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (name, 0, now_str()),
                )

                cur.execute(
                    """
                    INSERT INTO food_profiles(class_name, display_name, unit, shelf_life_days, storage_tip)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(class_name) DO UPDATE SET
                        display_name=excluded.display_name,
                        unit=excluded.unit,
                        shelf_life_days=excluded.shelf_life_days,
                        storage_tip=excluded.storage_tip
                    """,
                    (
                        name,
                        food_display_name(name),
                        food_unit(name),
                        food_shelf_life_days(name),
                        STORAGE_TIP.get(name, ""),
                    ),
                )

            self.conn.commit()

    def get_counts(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT class_name, count FROM inventory")
            rows = cur.fetchall()

        counts = empty_counts()
        for name, count in rows:
            if name in counts:
                counts[name] = normalize_count_value(count)
        return counts

    def set_count(self, class_name, count):
        normalized_count = normalize_count_value(count)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT count FROM inventory WHERE class_name=?", (class_name,))
            row = cur.fetchone()
            old_count = normalize_count_value(row[0]) if row else 0
            cur.execute(
                """
                UPDATE inventory
                SET count=?, updated_at=?
                WHERE class_name=?
                """,
                (normalized_count, now_str(), class_name),
            )
            should_clear_label = normalized_count <= 0
            if class_name in LABEL_SUPPORTED_CLASSES and old_count <= 0 < normalized_count:
                should_clear_label = True
            if should_clear_label:
                cur.execute(
                    "UPDATE label_records SET active=0 WHERE class_name=? AND active=1",
                    (class_name,),
                )
            self.conn.commit()

    def set_all_counts(self, counts):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT class_name, count FROM inventory")
            old_counts = {
                name: normalize_count_value(count)
                for name, count in cur.fetchall()
            }
            for name in CLASS_ORDER:
                normalized_count = normalize_count_value(counts.get(name, 0))
                old_count = normalize_count_value(old_counts.get(name, 0))
                cur.execute(
                    """
                    UPDATE inventory
                    SET count=?, updated_at=?
                    WHERE class_name=?
                    """,
                    (normalized_count, now_str(), name),
                )
                should_clear_label = normalized_count <= 0
                if name in LABEL_SUPPORTED_CLASSES and old_count <= 0 < normalized_count:
                    should_clear_label = True
                if should_clear_label:
                    cur.execute(
                        "UPDATE label_records SET active=0 WHERE class_name=? AND active=1",
                        (name,),
                    )
            self.conn.commit()

    def reset_zero(self):
        self.set_all_counts(empty_counts())
        self.clear_reminders()

    def log_event(self, class_name, delta, old_count, new_count):
        event_type = "in" if delta > 0 else "out"

        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO events(event_time, class_name, delta, old_count, new_count, event_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str(),
                    class_name,
                    normalize_count_value(delta),
                    normalize_count_value(old_count),
                    normalize_count_value(new_count),
                    event_type,
                ),
            )
            self.conn.commit()

    def get_events(self, limit=12):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT event_time, class_name, delta, old_count, new_count, event_type
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            return cur.fetchall()

    def get_last_update(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT MAX(updated_at) FROM inventory")
            row = cur.fetchone()
        if row and row[0]:
            return row[0]
        return "-"

    def get_inventory_items(self):
        counts = self.get_counts()
        items = []
        for name in CLASS_ORDER:
            items.append({
                "class_name": name,
                "display_name": food_display_name(name),
                "unit": food_unit(name),
                "count": normalize_count_value(counts.get(name, 0)),
                "shelf_life_days": food_shelf_life_days(name),
                "storage_tip": STORAGE_TIP.get(name, ""),
                "latest_label": self.get_latest_label_record(name),
            })
        return items

    def clear_reminders(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM reminders WHERE done=0")
            self.conn.commit()

    def set_default_reminders(self):
        items = [item for item in self.get_inventory_items() if item["count"] > 0]
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM reminders WHERE done=0")
            created = 0
            for item in items:
                remind_at = (
                    datetime.now() + timedelta(days=max(1, int(item["shelf_life_days"])))
                ).strftime("%Y-%m-%d 09:00:00")
                message = (
                    f"{item['display_name']}当前库存{format_food_count(item['class_name'], item['count'])}，"
                    f"建议{item['shelf_life_days']}天内食用。"
                )
                cur.execute(
                    """
                    INSERT INTO reminders(class_name, remind_at, message, done, created_at)
                    VALUES (?, ?, ?, 0, ?)
                    """,
                    (item["class_name"], remind_at, message, now_str()),
                )
                created += 1
            self.conn.commit()
            return created

    def get_reminders(self, limit=8):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT remind_at, class_name, message, done
                FROM reminders
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            return cur.fetchall()

    def save_label_record(self, record):
        created_at = record.get("created_at") or now_str()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE label_records SET active=0 WHERE class_name=? AND active=1",
                (record.get("class_name", ""),),
            )
            cur.execute(
                """
                INSERT INTO label_records(
                    class_name, product_name, brand, shelf_life_text,
                    production_date, expiry_date, nutrition_text,
                    calories, protein, fat, carbohydrate, sodium,
                    ocr_text, image_path, active, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    record.get("class_name", ""),
                    record.get("product_name", ""),
                    record.get("brand", ""),
                    record.get("shelf_life_text", ""),
                    record.get("production_date", ""),
                    record.get("expiry_date", ""),
                    record.get("nutrition_text", ""),
                    record.get("calories", ""),
                    record.get("protein", ""),
                    record.get("fat", ""),
                    record.get("carbohydrate", ""),
                    record.get("sodium", ""),
                    record.get("ocr_text", ""),
                    record.get("image_path", ""),
                    created_at,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_latest_label_record(self, class_name):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT id, class_name, product_name, brand, shelf_life_text,
                       production_date, expiry_date, nutrition_text,
                       calories, protein, fat, carbohydrate, sodium,
                       ocr_text, image_path, created_at
                FROM label_records
                WHERE class_name=? AND active=1
                ORDER BY id DESC
                LIMIT 1
                """,
                (class_name,),
            )
            row = cur.fetchone()

        if not row:
            return None

        keys = [
            "id",
            "class_name",
            "product_name",
            "brand",
            "shelf_life_text",
            "production_date",
            "expiry_date",
            "nutrition_text",
            "calories",
            "protein",
            "fat",
            "carbohydrate",
            "sodium",
            "ocr_text",
            "image_path",
            "created_at",
        ]
        record = dict(zip(keys, row))
        record["advice"] = build_label_advice(class_name, record)
        return record

    def close(self):
        with self.lock:
            self.conn.close()


# ============================================================
# Inventory assistant / cloud LLM
# ============================================================

class CloudLLMClient:
    def __init__(self, api_url="", api_key="", model="", timeout=12):
        self.api_url = api_url.strip()
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = timeout

    def available(self):
        return bool(self.api_url and self.model)

    def ask(self, question, inventory_items):
        if not self.available():
            return ""

        inventory_text = json.dumps(inventory_items, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是智能冰箱的中文语音助手。只能基于提供的库存 JSON 回答，"
                    "JSON 中 inventory 数组表示当前真实库存，只能把 inventory 中 count 大于 0 的食材说成还在冰箱里；"
                    "recent_events 只是历史事件，不能当作当前库存。不要编造不存在的库存；回答要短，适合语音播报。"
                ),
            },
            {
                "role": "user",
                "content": f"库存 JSON：{inventory_text}\n用户问题：{question}",
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8", errors="replace").strip()
            except Exception:
                error_body = ""
            if error_body:
                print(f"[WARN] LLM request failed: HTTP Error {e.code}: {e.reason}; body: {error_body[:500]}")
            else:
                print("[WARN] LLM request failed:", e)
            return ""
        except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as e:
            print("[WARN] LLM request failed:", e)
            return ""


def build_recent_event_payload(db, limit=6):
    events = []
    for event_time, class_name, delta, old_count, new_count, event_type in db.get_events(limit=limit):
        events.append({
            "event_time": event_time,
            "class_name": class_name,
            "display_name": food_display_name(class_name),
            "delta": normalize_count_value(delta),
            "old_count": normalize_count_value(old_count),
            "new_count": normalize_count_value(new_count),
            "event_type": event_type,
        })
    return events


def build_cloud_analysis_payload(db):
    current_inventory = [
        item for item in db.get_inventory_items()
        if normalize_count_value(item.get("count", 0)) > 0
    ]
    return {
        "generated_at": now_str(),
        "inventory": current_inventory,
        "recent_events": build_recent_event_payload(db),
    }


class CloudAnalysisManager:
    def __init__(self, db, speaker, llm_client=None, debounce_seconds=1.0):
        self.db = db
        self.speaker = speaker
        self.llm_client = llm_client or CloudLLMClient()
        self.debounce_seconds = max(0.0, float(debounce_seconds))
        self.lock = threading.RLock()
        self.q = queue.Queue(maxsize=1)
        self.request_seq = 0
        self.latest_analysis_request_id = 0
        self.latest_voice_request_id = 0
        self.status = "idle"
        self.status_text = (
            "等待云端分析" if self.llm_client.available() else "云端未启用，本地回退可用"
        )
        self.source = ""
        self.answer = "暂无云端建议。"
        self.voice_status = "idle"
        self.voice_question = ""
        self.voice_answer = ""
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def get_state(self):
        with self.lock:
            return {
                "status": self.status,
                "status_text": self.status_text,
                "source": self.source,
                "answer": self.answer,
                "voice_status": self.voice_status,
                "voice_question": self.voice_question,
                "voice_answer": self.voice_answer,
            }

    def clear(self, status_text="等待云端分析"):
        with self.lock:
            self.status = "idle"
            self.status_text = status_text
            self.source = ""
            self.answer = "暂无云端建议。"
            self.voice_status = "idle"
            self.voice_question = ""
            self.voice_answer = ""

    def request_analysis(self, reason="manual"):
        return self._enqueue(
            kind="analysis",
            reason=reason,
            question=CLOUD_ANALYSIS_QUESTION,
            speak=False,
        )

    def request_voice_question(self, question=SIMULATED_VOICE_QUESTION):
        return self._enqueue(
            kind="voice",
            reason="voice_question",
            question=question,
            speak=True,
        )

    def speak_latest(self):
        state = self.get_state()
        if state["answer"] and state["status"] in {"done", "fallback"}:
            self.speaker.say(state["answer"])
            return state["answer"]
        text = "请先点击云分析，生成云端建议。"
        self.speaker.say(text)
        return text

    def _enqueue(self, kind, reason, question, speak=False):
        payload = build_cloud_analysis_payload(self.db)
        with self.lock:
            self.request_seq += 1
            request_id = self.request_seq
            if kind == "analysis":
                self.latest_analysis_request_id = request_id
                self.status = "queued"
                self.status_text = "等待云端分析..."
            else:
                self.latest_voice_request_id = request_id
                self.voice_status = "queued"
                self.voice_question = question
                self.voice_answer = "等待云端回答..."

        item = {
            "request_id": request_id,
            "kind": kind,
            "reason": reason,
            "question": question,
            "payload": payload,
            "speak": speak,
        }
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                pass
            self.q.put_nowait(item)
        print(f"[CLOUD] queued {kind}: {reason}, request_id={request_id}")
        return request_id

    def _worker(self):
        while True:
            item = self.q.get()
            try:
                if self.debounce_seconds > 0:
                    time.sleep(self.debounce_seconds)
                    while True:
                        try:
                            newer_item = self.q.get_nowait()
                            self.q.task_done()
                            item = newer_item
                        except queue.Empty:
                            break
                self._run_request(item)
            except Exception as e:
                print("[WARN] cloud analysis failed:", e)
            finally:
                self.q.task_done()

    def _run_request(self, item):
        request_id = item["request_id"]
        kind = item["kind"]
        question = item["question"]
        payload = item["payload"]
        speak = item["speak"]

        with self.lock:
            if kind == "analysis":
                self.status = "running"
                self.status_text = "云端分析中..."
            else:
                self.voice_status = "running"
                self.voice_answer = "云端回答生成中..."

        answer = ""
        if self.llm_client.available():
            answer = self.llm_client.ask(question, payload)

        if answer:
            source = "cloud"
            status = "done"
            status_text = "云端分析完成"
        else:
            source = "local_fallback"
            status = "fallback"
            status_text = (
                "请求失败，已使用本地建议"
                if self.llm_client.available()
                else "云端未启用，已使用本地建议"
            )
            answer = build_advice_answer(self.db)

        should_speak = False
        with self.lock:
            if kind == "analysis":
                if request_id != self.latest_analysis_request_id:
                    print(f"[CLOUD] skip stale analysis result: {request_id}")
                    return
                self.status = status
                self.status_text = status_text
                self.source = source
                self.answer = answer
            else:
                if request_id != self.latest_voice_request_id:
                    print(f"[CLOUD] skip stale voice result: {request_id}")
                    return
                self.voice_status = status
                self.voice_question = question
                self.voice_answer = answer
                should_speak = speak

        print(f"[CLOUD] {kind} {status}: {answer}")
        if should_speak:
            self.speaker.say(answer)


class FridgeAssistant:
    def __init__(self, db, speaker, llm_client=None):
        self.db = db
        self.speaker = speaker
        self.llm_client = llm_client or CloudLLMClient()
        self.latest_question = ""
        self.latest_answer = "助手已就绪。按 A 查询库存，按 D 获取食用建议，按 R 设置提醒。"
        self.busy = False
        self.q = queue.Queue(maxsize=3)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def ask(self, question):
        self.latest_question = question
        try:
            self.q.put_nowait(question)
        except queue.Full:
            self.latest_answer = "助手正在处理上一条请求，请稍后再试。"
            print("[ASSISTANT]", self.latest_answer)

    def _worker(self):
        while True:
            question = self.q.get()
            self.busy = True
            try:
                answer = self._answer(question)
                self.latest_answer = answer
                print("[ASSISTANT]", answer)
                self.speaker.say(answer)
            except Exception as e:
                self.latest_answer = "助手处理失败，请查看终端日志。"
                print("[WARN] assistant failed:", e)
            finally:
                self.busy = False
                self.q.task_done()

    def _answer(self, question):
        normalized = question.strip().lower()
        if any(word in normalized for word in ["提醒", "remind", "alarm"]):
            return build_reminder_answer(self.db)

        if any(word in normalized for word in ["建议", "多久", "食用", "保质", "吃", "advice"]):
            return build_advice_answer(self.db)

        if any(word in normalized for word in ["库存", "有什么", "哪些", "现在", "inventory", "stock"]):
            return build_inventory_answer(self.db)

        llm_answer = self.llm_client.ask(question, self.db.get_inventory_items())
        if llm_answer:
            return llm_answer

        return build_inventory_answer(self.db)


# ============================================================
# RKNN preprocess / postprocess
# ============================================================

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


def class_score_by_name(class_scores, classes, class_name):
    try:
        idx = list(classes).index(class_name)
    except ValueError:
        return None
    if idx < 0 or idx >= len(class_scores):
        return None
    return float(class_scores[idx])


def is_ambiguous_milk_coke(class_name, class_scores, classes):
    if MILK_COKE_MIN_MARGIN <= 0:
        return False
    if class_name not in ["milk_box", "coke_can"]:
        return False

    other = "coke_can" if class_name == "milk_box" else "milk_box"
    this_score = class_score_by_name(class_scores, classes, class_name)
    other_score = class_score_by_name(class_scores, classes, other)
    if this_score is None or other_score is None:
        return False
    return (this_score - other_score) < MILK_COKE_MIN_MARGIN


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
    cls_scores = cls_scores[keep]

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
    valid_class_scores = []

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
        valid_class_scores.append(cls_scores[i])

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

        if cname not in CLASS_ORDER:
            continue
        class_scores = valid_class_scores[i]
        if is_ambiguous_milk_coke(cname, class_scores, classes):
            continue

        results.append({
            "class_id": cid,
            "class_name": cname,
            "score": valid_scores[i],
            "box": valid_xyxy[i],
            "class_margin": float(
                class_scores[cid] - max(
                    [score for idx, score in enumerate(class_scores) if idx != cid] or [0.0]
                )
            ),
        })

    return results


def crop_inner_roi(frame_bgr, box, shrink_ratio=0.0):
    if frame_bgr is None or box is None:
        return None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    shrink_ratio = max(0.0, min(0.4, float(shrink_ratio)))
    dx = int((x2 - x1) * shrink_ratio)
    dy = int((y2 - y1) * shrink_ratio)
    x1 += dx
    x2 -= dx
    y1 += dy
    y2 -= dy
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def blueberry_hsv_area_ratio(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUEBERRY_HSV_LOWER, BLUEBERRY_HSV_UPPER)
    return float(cv2.countNonZero(mask)) / float(mask.size)


def blueberry_binary_dark_area_ratio(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if BLUEBERRY_BINARY_THRESHOLD <= 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, BLUEBERRY_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return float(cv2.countNonZero(mask)) / float(mask.size)


def blueberry_area_ratio(roi):
    method = BLUEBERRY_AREA_METHOD.lower().strip()
    if method in ["binary", "binary_dark", "gray", "threshold"]:
        return blueberry_binary_dark_area_ratio(roi)
    if method in ["hsv", "color", "blue_purple"]:
        return blueberry_hsv_area_ratio(roi)

    binary_ratio = blueberry_binary_dark_area_ratio(roi)
    hsv_ratio = blueberry_hsv_area_ratio(roi)
    return max(binary_ratio, hsv_ratio)


def estimate_blueberry_box_count(frame_bgr, box):
    roi = crop_inner_roi(frame_bgr, box, BLUEBERRY_ROI_SHRINK)
    if roi is None:
        return 1.0

    color_ratio = blueberry_hsv_area_ratio(roi)
    if BLUEBERRY_COLOR_GUARD_RATIO > 0 and color_ratio < BLUEBERRY_COLOR_GUARD_RATIO:
        return 0.0

    area_ratio = blueberry_area_ratio(roi)

    if area_ratio >= BLUEBERRY_FULL_RATIO:
        return 1.0
    if area_ratio >= BLUEBERRY_HALF_RATIO:
        return 0.5
    return 0.0


def filter_detections_by_content(results, frame_bgr=None):
    filtered = []
    for det in results:
        if det.get("class_name") == "blueberry_box":
            if estimate_blueberry_box_count(frame_bgr, det.get("box")) <= 0:
                continue
        filtered.append(det)
    return filtered


def count_detections(results, frame_bgr=None):
    counts = empty_counts()
    for det in results:
        name = det["class_name"]
        if name in counts:
            if name == "blueberry_box":
                counts[name] = normalize_count_value(
                    counts[name] + estimate_blueberry_box_count(frame_bgr, det.get("box"))
                )
            else:
                counts[name] += 1
    return counts


def assess_frame_quality(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons = []
    codes = []
    if brightness < LOW_LIGHT_MEAN:
        reasons.append("光照不足")
        codes.append("low_light")
    if contrast < LOW_CONTRAST_STD:
        reasons.append("低对比度/疑似遮挡")
        codes.append("low_contrast")
    if blur_var < BLUR_VAR_THRES and contrast < 35.0:
        reasons.append("画面模糊")
        codes.append("blur")

    freeze = len(reasons) > 0
    return {
        "freeze": freeze,
        "brightness": brightness,
        "contrast": contrast,
        "blur_var": blur_var,
        "message": "、".join(reasons) if reasons else "画面正常",
        "code": "+".join(codes) if codes else "OK",
    }


def odd_kernel_size(value, minimum=3):
    value = max(int(value), int(minimum))
    if value % 2 == 0:
        value += 1
    return value


def suppress_plastic_glare(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.where(
        (v >= PREPROCESS_GLARE_V_THRESHOLD) & (s <= PREPROCESS_GLARE_S_THRESHOLD),
        255,
        0,
    ).astype(np.uint8)

    if cv2.countNonZero(mask) == 0:
        return frame_bgr

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    blur_kernel = odd_kernel_size(PREPROCESS_GLARE_BLUR_KERNEL, minimum=3)
    blurred = cv2.medianBlur(frame_bgr, blur_kernel)

    out = frame_bgr.copy()
    out[mask > 0] = blurred[mask > 0]
    return out


def apply_lab_clahe(frame_bgr):
    if PREPROCESS_CLAHE_CLIP_LIMIT <= 0:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    tile = max(2, int(PREPROCESS_CLAHE_TILE_GRID))
    clahe = cv2.createCLAHE(
        clipLimit=float(PREPROCESS_CLAHE_CLIP_LIMIT),
        tileGridSize=(tile, tile),
    )
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma(frame_bgr):
    gamma = float(PREPROCESS_GAMMA)
    if gamma <= 0 or abs(gamma - 1.0) < 1e-6:
        return frame_bgr

    table = np.array([
        np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(frame_bgr, table)


def boost_saturation(frame_bgr):
    gain = float(PREPROCESS_SATURATION_GAIN)
    if abs(gain - 1.0) < 1e-6:
        return frame_bgr

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def sharpen_frame(frame_bgr):
    amount = float(PREPROCESS_SHARPEN_AMOUNT)
    if amount <= 0:
        return frame_bgr

    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), 1.0)
    return cv2.addWeighted(frame_bgr, 1.0 + amount, blurred, -amount, 0)


def enhance_plastic_wrap_frame(frame_bgr):
    out = frame_bgr.copy()
    if PREPROCESS_GLARE_SUPPRESS:
        out = suppress_plastic_glare(out)
    out = apply_lab_clahe(out)
    out = apply_gamma(out)
    out = boost_saturation(out)
    out = cv2.convertScaleAbs(
        out,
        alpha=float(PREPROCESS_CONTRAST_ALPHA),
        beta=float(PREPROCESS_BRIGHTNESS_BETA),
    )
    out = sharpen_frame(out)
    return out


def preprocess_frame_for_inference(frame_bgr):
    if not PREPROCESS_ENABLED:
        return frame_bgr

    method = PREPROCESS_METHOD.lower().strip()
    if method in ["none", "off", "raw"]:
        return frame_bgr
    if method in ["plastic", "plastic_wrap", "film", "transparent_film"]:
        return enhance_plastic_wrap_frame(frame_bgr)
    if method in ["clahe", "contrast"]:
        return apply_lab_clahe(frame_bgr.copy())

    return enhance_plastic_wrap_frame(frame_bgr)


# ============================================================
# Stable counter
# ============================================================

class StableCounter:
    def __init__(self, window_size, votes):
        self.window_size = window_size
        self.votes = votes
        self.history = deque(maxlen=window_size)

    def update(self, counts):
        sig = counts_to_signature(counts)
        self.history.append(sig)

        if len(self.history) < self.window_size:
            return None, 0

        counter = Counter(self.history)
        best_sig, best_votes = counter.most_common(1)[0]

        if best_votes >= self.votes:
            return signature_to_counts(best_sig), best_votes

        return None, best_votes


# ============================================================
# Camera
# ============================================================

def video_device_sort_key(device):
    name = Path(str(device)).name
    suffix = name.replace("video", "", 1)
    if suffix.isdigit():
        return int(suffix)
    return 9999


def list_video_devices():
    return [
        str(path)
        for path in sorted(Path("/dev").glob("video*"), key=lambda p: video_device_sort_key(p))
        if path.name.startswith("video") and path.name.replace("video", "", 1).isdigit()
    ]


def camera_candidates(device):
    requested = str(device).strip()
    detected = list_video_devices()
    if requested.lower() == "auto":
        return detected
    candidates = [requested]
    candidates.extend(path for path in detected if path != requested)
    return candidates


def open_single_camera(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print("[INFO] camera:", device)
    print("[INFO] width:", real_w)
    print("[INFO] height:", real_h)
    print("[INFO] fps:", real_fps)
    print("[INFO] fourcc:", fourcc_str)

    return cap


def open_camera(device):
    attempted = []
    for candidate in camera_candidates(device):
        attempted.append(candidate)
        cap = open_single_camera(candidate)
        if cap is not None:
            if candidate != device:
                print(f"[INFO] selected fallback camera: {candidate}")
            return cap
        print(f"[WARN] failed to open camera candidate: {candidate}")

    raise RuntimeError(
        "failed to open camera. "
        f"requested={device}, attempted={attempted}. "
        "Run: ls -l /dev/video* && v4l2-ctl --list-devices"
    )


# ============================================================
# Inventory event logic
# ============================================================

def apply_inventory_change(db, speaker, stable_counts, last_event_time, cloud_analysis=None):
    db_counts = db.get_counts()
    now = time.time()
    changed = False

    for name in CLASS_ORDER:
        old_count = normalize_count_value(db_counts.get(name, 0))
        new_count = normalize_count_value(stable_counts.get(name, 0))
        delta = normalize_count_value(new_count - old_count)

        if delta == 0:
            continue

        last_t = last_event_time.get(name, 0.0)
        if now - last_t < EVENT_COOLDOWN:
            continue

        db.set_count(name, new_count)
        db.log_event(name, delta, old_count, new_count)
        last_event_time[name] = now
        changed = True
        if new_count <= 0:
            clear_label_ui_state(name, "对应食材已取出，标签信息已清空。")

        event_type = "IN" if delta > 0 else "OUT"
        print(f"[EVENT] {event_type} {name}: {old_count} -> {new_count}, delta={delta}")

        speaker.say(build_speak_text(name, delta, new_count))

    if changed and cloud_analysis is not None:
        cloud_analysis.request_analysis(reason="inventory_change")


# ============================================================
# UI drawing
# ============================================================

FONT_CACHE = {}
TEXT_SIZE_CACHE = {}
TEXT_RENDER_CACHE = {}
TEXT_RENDER_CACHE_MAX = 512


def has_non_ascii(text):
    return any(ord(ch) > 127 for ch in str(text))


def find_font_path():
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return ""


CHINESE_FONT_PATH = find_font_path()


def get_pil_font(scale):
    if ImageFont is None:
        return None

    size = max(12, int(scale * 32))
    if size in FONT_CACHE:
        return FONT_CACHE[size]

    try:
        if CHINESE_FONT_PATH:
            font = ImageFont.truetype(CHINESE_FONT_PATH, size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    FONT_CACHE[size] = font
    return font


def pil_text_bbox(font, text):
    if hasattr(font, "getbbox"):
        return font.getbbox(text)
    if hasattr(font, "getsize"):
        w, h = font.getsize(text)
        return (0, 0, w, h)
    return (0, 0, 0, 0)


def measure_text(text, scale=0.6, thickness=1):
    text = str(text)
    if has_non_ascii(text) and ImageFont is not None:
        cache_key = (text, int(scale * 1000), int(thickness))
        cached = TEXT_SIZE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        font = get_pil_font(scale)
        if font is not None:
            bbox = pil_text_bbox(font, text)
            size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            TEXT_SIZE_CACHE[cache_key] = size
            return size

    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]


def render_text_sprite(text, scale=0.6, color=(255, 255, 255)):
    if Image is None or ImageDraw is None:
        return None

    text = str(text)
    font = get_pil_font(scale)
    if font is None:
        return None

    key = (text, int(scale * 1000), tuple(int(v) for v in color))
    cached = TEXT_RENDER_CACHE.get(key)
    if cached is not None:
        return cached

    bbox = pil_text_bbox(font, text)
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    pil_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pil_img)
    draw.text(
        (1 - bbox[0], 1 - bbox[1]),
        text,
        font=font,
        fill=(int(color[2]), int(color[1]), int(color[0]), 255),
    )

    rgba = np.asarray(pil_img)
    bgr = rgba[:, :, :3][:, :, ::-1].copy()
    alpha = rgba[:, :, 3].copy()
    sprite = (bgr, alpha)

    if len(TEXT_RENDER_CACHE) >= TEXT_RENDER_CACHE_MAX:
        TEXT_RENDER_CACHE.clear()
    TEXT_RENDER_CACHE[key] = sprite
    return sprite


def blit_text_sprite(img, sprite, pos):
    bgr, alpha = sprite
    x, y = int(pos[0]), int(pos[1])
    h, w = alpha.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    src = bgr[sy1:sy2, sx1:sx2].astype(np.uint16)
    a = alpha[sy1:sy2, sx1:sx2].astype(np.uint16)[:, :, None]
    dst = img[y1:y2, x1:x2].astype(np.uint16)
    img[y1:y2, x1:x2] = ((src * a + dst * (255 - a)) // 255).astype(np.uint8)


def draw_round_rect(img, x1, y1, x2, y2, color, thickness=-1, radius=14):
    # 轻量近似圆角矩形
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def put_text(img, text, pos, scale=0.6, color=(255, 255, 255), thickness=1):
    text = str(text)
    if has_non_ascii(text) and Image is not None and ImageDraw is not None:
        sprite = render_text_sprite(text, scale, color)
        if sprite is not None:
            _, alpha = sprite
            blit_text_sprite(img, sprite, (int(pos[0]), int(pos[1]) - alpha.shape[0]))
            return

    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def register_button(x1, y1, x2, y2, label, action, active=False):
    UI_STATE["buttons"].append({
        "rect": (x1, y1, x2, y2),
        "label": label,
        "action": action,
        "active": active,
    })


def draw_button(img, x1, y1, x2, y2, label, action, active=False):
    register_button(x1, y1, x2, y2, label, action, active)

    color = (70, 140, 255) if active else (55, 65, 80)
    border = (120, 180, 255) if active else (90, 100, 115)

    draw_round_rect(img, x1, y1, x2, y2, color, -1, radius=10)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)

    text_size = measure_text(label, 0.55, 1)
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = y1 + (y2 - y1 + text_size[1]) // 2
    put_text(img, label, (tx, ty), 0.55, (255, 255, 255), 1)


def draw_header(img, title, subtitle):
    UI_STATE["buttons"] = []

    cv2.rectangle(img, (0, 0), (UI_W, 64), (18, 24, 32), -1)
    put_text(img, title, (20, 26), 0.72, (255, 255, 255), 2)
    put_text(img, subtitle, (20, 50), 0.45, (160, 180, 200), 1)

    page = UI_STATE["page"]

    draw_button(img, 430, 16, 490, 48, "识别", PAGE_DETECT, page == PAGE_DETECT)
    draw_button(img, 500, 16, 560, 48, "库存", PAGE_INVENTORY, page == PAGE_INVENTORY)
    draw_button(img, 570, 16, 630, 48, "标签", PAGE_LABEL, page == PAGE_LABEL)
    draw_button(img, 640, 16, 700, 48, "事件", PAGE_EVENTS, page == PAGE_EVENTS)
    draw_button(img, 710, 16, 770, 48, "助手", PAGE_ASSISTANT, page == PAGE_ASSISTANT)
    draw_button(img, 780, 16, 840, 48, "清空", "reset", False)
    draw_button(img, 850, 16, 920, 48, "初始化", "init", False)
    draw_button(img, 930, 16, 990, 48, "退出", "quit", False)


def draw_status_bar(img, fps, infer_ms, stable_votes, frame_quality=None):
    cv2.rectangle(img, (0, UI_H - 42), (UI_W, UI_H), (18, 24, 32), -1)
    quality_code = "OK"
    if frame_quality is not None:
        quality_code = frame_quality.get("code", "OK")
    put_text(
        img,
        f"FPS {fps:.1f} | Infer {infer_ms:.1f} ms | Stable {stable_votes}/{STABLE_WINDOW} | Quality {quality_code} | q/Exit | 1 Detect 2 Stock 3 Events 4 AI 5/l Label | c Capture s Save x Clear | z Reset i Init",
        (18, UI_H - 16),
        0.48,
        (180, 220, 255),
        1,
    )


def fit_image_to_box(img, box_w, box_h):
    h, w = img.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def draw_detection_boxes(frame, detections):
    vis = frame.copy()
    for det in detections:
        name = det["class_name"]
        color = CLASS_COLOR.get(name, (0, 255, 0))
        x1, y1, x2, y2 = det["box"]
        label = f"{CLASS_DISPLAY_NAME.get(name, name)} {det['score']:.2f}"
        label_w, label_h = measure_text(label, 0.5, 1)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - 26)), (x1 + max(110, label_w + 12), y1), color, -1)
        put_text(vis, label, (x1 + 4, max(16, y1 - 6)), 0.5, (255, 255, 255), 1)
    return vis


def draw_inventory_small_cards(img, x, y, db_counts, stable_counts):
    card_w = 245
    card_h = 58
    gap = 10

    for idx, name in enumerate(CLASS_ORDER):
        yy = y + idx * (card_h + gap)
        color = CLASS_COLOR.get(name, (80, 160, 255))
        db_v = db_counts.get(name, 0)
        st_v = stable_counts.get(name, "-") if stable_counts is not None else "-"

        draw_round_rect(img, x, yy, x + card_w, yy + card_h, (32, 39, 50), -1, 12)
        cv2.circle(img, (x + 24, yy + 29), 12, color, -1)

        put_text(img, CLASS_DISPLAY_NAME.get(name, name), (x + 46, yy + 24), 0.55, (240, 245, 255), 1)
        db_text = format_food_count(name, db_v)
        st_text = "-" if st_v == "-" else format_food_count(name, st_v)
        put_text(img, f"库存 {db_text}   稳定 {st_text}", (x + 46, yy + 47), 0.45, (155, 175, 195), 1)


def draw_detect_page(frame, detections, detect_counts, stable_counts, db_counts, fps, infer_ms, stable_votes, frame_quality=None):
    canvas = np.full((UI_H, UI_W, 3), (12, 17, 24), dtype=np.uint8)

    draw_header(canvas, "智能冰箱食材识别", "实时检测、稳定计数、自动更新库存")

    video = draw_detection_boxes(frame, detections)
    video_box = fit_image_to_box(video, 700, 480)

    draw_round_rect(canvas, 18, 82, 728, 572, (26, 32, 42), -1, 18)
    canvas[88:88 + 480, 23:23 + 700] = video_box

    draw_round_rect(canvas, 750, 82, 982, 572, (22, 28, 38), -1, 18)
    put_text(canvas, "实时库存", (770, 115), 0.68, (255, 255, 255), 2)
    put_text(canvas, "数据库数量 / 稳定识别数量", (770, 140), 0.42, (150, 170, 190), 1)
    if frame_quality is not None and frame_quality.get("freeze"):
        put_text(canvas, f"已暂停更新：{frame_quality.get('message', '')}", (770, 158), 0.42, (80, 190, 255), 1)

    draw_inventory_small_cards(canvas, 768, 160, db_counts, stable_counts)

    draw_status_bar(canvas, fps, infer_ms, stable_votes, frame_quality)
    return canvas


def draw_inventory_page(db, detect_counts, stable_counts, db_counts, fps, infer_ms, stable_votes, frame_quality=None):
    canvas = np.full((UI_H, UI_W, 3), (12, 17, 24), dtype=np.uint8)

    draw_header(canvas, "库存数据库", "点击食材标签查看保质期、营养成分和食用建议")

    put_text(canvas, f"上次更新：{db.get_last_update()}", (28, 96), 0.58, (170, 190, 210), 1)

    selected_class = UI_STATE.get("inventory_selected_class", "apple")
    if selected_class not in CLASS_ORDER:
        selected_class = CLASS_ORDER[0]
        UI_STATE["inventory_selected_class"] = selected_class

    start_x = 28
    start_y = 118
    card_w = 145
    card_h = 88
    gap_x = 14
    for idx, name in enumerate(CLASS_ORDER):
        x = start_x + idx * (card_w + gap_x)
        y = start_y

        color = CLASS_COLOR.get(name, (80, 160, 255))
        db_v = db_counts.get(name, 0)
        st_v = stable_counts.get(name, "-") if stable_counts is not None else "-"
        d_v = detect_counts.get(name, 0)
        active = name == selected_class
        bg = (40, 55, 72) if active else (28, 35, 48)
        border = (115, 180, 255) if active else (55, 70, 88)

        register_button(x, y, x + card_w, y + card_h, CLASS_DISPLAY_NAME.get(name, name), f"inventory_select_{name}", active)
        draw_round_rect(canvas, x, y, x + card_w, y + card_h, bg, -1, 12)
        cv2.rectangle(canvas, (x, y), (x + card_w, y + card_h), border, 1)
        cv2.circle(canvas, (x + 24, y + 28), 12, color, -1)

        put_text(canvas, CLASS_DISPLAY_NAME.get(name, name), (x + 42, y + 30), 0.5, (250, 250, 255), 1)
        put_text(canvas, f"{format_count_value(db_v)}{food_unit(name)}", (x + 18, y + 62), 0.72, (255, 255, 255), 2)

        st_text = "-" if st_v == "-" else format_count_value(st_v)
        put_text(canvas, f"检{format_count_value(d_v)} 稳{st_text}", (x + 82, y + 65), 0.38, (165, 190, 210), 1)

    detail = build_inventory_detail(db, selected_class)
    detail_x1 = 28
    detail_y1 = 230
    detail_x2 = UI_W - 28
    detail_y2 = 548
    draw_round_rect(canvas, detail_x1, detail_y1, detail_x2, detail_y2, (22, 28, 38), -1, 16)
    status_color = {
        "ok": (120, 220, 170),
        "missing_label": (80, 190, 255),
        "no_stock": (160, 180, 200),
    }.get(detail.get("status"), (160, 180, 200))
    status_text = {
        "ok": "营养信息已就绪",
        "missing_label": "等待标签录入",
        "no_stock": "未入库",
    }.get(detail.get("status"), detail.get("status", "-"))
    put_text(canvas, detail["title"], (50, 268), 0.8, (255, 255, 255), 2)
    put_text(canvas, f"状态：{status_text}", (180, 268), 0.46, status_color, 1)

    left_x1, left_y1, left_x2, left_y2 = 52, 292, 470, 476
    right_x1, right_y1, right_x2, right_y2 = 505, 292, 950, 476
    put_text(canvas, "基础信息", (left_x1, left_y1), 0.6, (255, 255, 255), 1)
    put_text(canvas, "营养配料表", (right_x1, right_y1), 0.6, (255, 255, 255), 1)
    draw_key_value_block(canvas, left_x1, left_y1 + 32, detail.get("metadata", []), key_w=96, max_rows=6)
    draw_nutrition_table(canvas, right_x1, right_y1 + 20, right_x2, detail.get("nutrition_rows", []))

    raw_lines = []
    for line in detail.get("lines", []):
        if "：" in str(line) and detail.get("status") == "ok":
            continue
        raw_lines.extend(wrap_cn_text(line, 56))
    scroll = max(0, int(UI_STATE.get("inventory_detail_scroll", 0)))
    max_visible = 3
    max_scroll = max(0, len(raw_lines) - max_visible)
    if scroll > max_scroll:
        scroll = max_scroll
        UI_STATE["inventory_detail_scroll"] = scroll

    y = 500
    for line in raw_lines[scroll:scroll + max_visible]:
        put_text(canvas, line, (52, y), 0.48, (225, 238, 248), 1)
        y += 23

    if max_scroll > 0:
        put_text(canvas, f"{scroll + 1}/{max_scroll + 1}", (852, 268), 0.44, (170, 190, 210), 1)
        draw_button(canvas, 885, 244, 935, 276, "上页", "inventory_scroll_up", False)
        draw_button(canvas, 940, 244, 990, 276, "下页", "inventory_scroll_down", False)

    draw_status_bar(canvas, fps, infer_ms, stable_votes, frame_quality)
    return canvas


def draw_events_page(db, fps, infer_ms, stable_votes, frame_quality=None):
    canvas = np.full((UI_H, UI_W, 3), (12, 17, 24), dtype=np.uint8)

    draw_header(canvas, "事件记录", "由库存数量变化自动生成入库/出库事件")

    events = db.get_events(limit=12)

    x = 40
    y = 105

    header_color = (35, 45, 60)
    draw_round_rect(canvas, x, y, UI_W - 40, y + 42, header_color, -1, 12)

    put_text(canvas, "时间", (x + 25, y + 27), 0.55, (210, 225, 240), 1)
    put_text(canvas, "食材", (x + 230, y + 27), 0.55, (210, 225, 240), 1)
    put_text(canvas, "变化", (x + 430, y + 27), 0.55, (210, 225, 240), 1)
    put_text(canvas, "原值 -> 现值", (x + 610, y + 27), 0.55, (210, 225, 240), 1)
    put_text(canvas, "类型", (x + 805, y + 27), 0.55, (210, 225, 240), 1)

    y += 56

    if not events:
        put_text(canvas, "暂无事件记录。", (60, 200), 0.8, (180, 200, 220), 2)
    else:
        for idx, row in enumerate(events):
            event_time, class_name, delta, old_count, new_count, event_type = row
            yy = y + idx * 36
            bg = (25, 32, 43) if idx % 2 == 0 else (30, 38, 50)
            draw_round_rect(canvas, x, yy - 24, UI_W - 40, yy + 8, bg, -1, 8)

            t = event_time.split(" ")[-1]
            display = CLASS_DISPLAY_NAME.get(class_name, class_name)
            delta_text = format_signed_count_value(delta)
            color = (80, 220, 120) if delta > 0 else (80, 160, 255)

            put_text(canvas, t, (x + 25, yy), 0.52, (235, 240, 245), 1)
            put_text(canvas, display, (x + 230, yy), 0.52, (235, 240, 245), 1)
            put_text(canvas, delta_text, (x + 430, yy), 0.52, color, 2)
            put_text(
                canvas,
                f"{format_count_value(old_count)} -> {format_count_value(new_count)}",
                (x + 610, yy),
                0.52,
                (235, 240, 245),
                1,
            )
            type_text = "入库" if event_type == "in" else "出库"
            put_text(canvas, type_text, (x + 805, yy), 0.52, color, 2)

    draw_status_bar(canvas, fps, infer_ms, stable_votes, frame_quality)
    return canvas


def wrap_cn_text(text, max_chars):
    lines = []
    for paragraph in str(text).splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:
            split_at = max_chars
            for marker in ["。", "；", "，", "、", " "]:
                idx = paragraph.rfind(marker, 0, max_chars)
                if idx > 6:
                    split_at = idx + 1
                    break
            lines.append(paragraph[:split_at])
            paragraph = paragraph[split_at:].strip()
        if paragraph:
            lines.append(paragraph)
    return lines or [""]


def draw_key_value_block(canvas, x, y, rows, key_w=96, max_rows=7):
    yy = y
    for key, value in rows[:max_rows]:
        put_text(canvas, f"{key}：", (x, yy), 0.5, (170, 195, 215), 1)
        for idx, line in enumerate(wrap_cn_text(value, 18)[:2]):
            put_text(canvas, line, (x + key_w, yy + idx * 22), 0.5, (235, 242, 248), 1)
        yy += 44 if len(str(value)) > 18 else 28
    return yy


def draw_nutrition_table(canvas, x1, y1, x2, rows):
    header_h = 32
    row_h = 32
    draw_round_rect(canvas, x1, y1, x2, y1 + header_h, (38, 48, 62), -1, 8)
    put_text(canvas, "项目", (x1 + 18, y1 + 22), 0.5, (230, 240, 250), 1)
    put_text(canvas, "数值", (x1 + 150, y1 + 22), 0.5, (230, 240, 250), 1)
    if not rows:
        put_text(canvas, "暂无营养成分表。", (x1 + 18, y1 + 62), 0.52, (170, 190, 210), 1)
        return
    yy = y1 + header_h
    for idx, (key, value) in enumerate(rows[:7]):
        bg = (28, 36, 48) if idx % 2 == 0 else (24, 31, 42)
        cv2.rectangle(canvas, (x1, yy), (x2, yy + row_h), bg, -1)
        cv2.line(canvas, (x1 + 132, yy), (x1 + 132, yy + row_h), (55, 68, 84), 1)
        put_text(canvas, str(key), (x1 + 18, yy + 22), 0.48, (215, 230, 242), 1)
        put_text(canvas, str(value), (x1 + 150, yy + 22), 0.48, (245, 248, 252), 1)
        yy += row_h


def draw_label_page(frame, db, fps, infer_ms, stable_votes, frame_quality=None):
    canvas = np.full((UI_H, UI_W, 3), (12, 17, 24), dtype=np.uint8)
    draw_header(canvas, "包装标签录入", "抓拍标签、解析保质期和营养成分")

    target = UI_STATE.get("label_target", "milk_box")
    record = UI_STATE.get("label_record") or db.get_latest_label_record(target)
    capture_path = UI_STATE.get("label_capture_path", "")
    status = UI_STATE.get("label_status", "请将包装标签对准取景框。")

    draw_round_rect(canvas, 18, 82, 618, 560, (26, 32, 42), -1, 18)
    video_box = fit_image_to_box(frame, 580, 430)
    canvas[105:105 + 430, 28:28 + 580] = video_box
    cv2.rectangle(canvas, (190, 190), (450, 380), (80, 220, 255), 2)
    put_text(canvas, "请将包装标签对准取景框", (205, 178), 0.58, (180, 230, 255), 1)

    draw_round_rect(canvas, 638, 82, 982, 560, (22, 28, 38), -1, 18)
    put_text(canvas, "录入对象", (660, 118), 0.68, (255, 255, 255), 2)
    draw_button(canvas, 660, 138, 745, 173, "牛奶", "label_target_milk", target == "milk_box")
    draw_button(canvas, 760, 138, 845, 173, "可乐", "label_target_coke", target == "coke_can")

    put_text(canvas, f"状态：{status}", (660, 210), 0.5, (210, 230, 245), 1)
    capture_name = Path(capture_path).name if capture_path else "尚未抓拍"
    put_text(canvas, f"图片：{capture_name}", (660, 238), 0.46, (150, 175, 195), 1)

    y = 282
    if record:
        put_text(canvas, f"产品：{record.get('product_name', '-')}", (660, y), 0.54, (245, 250, 255), 1)
        y += 28
        put_text(canvas, f"品牌：{record.get('brand', '-')}", (660, y), 0.5, (210, 230, 245), 1)
        y += 26
        put_text(canvas, f"保质期：{record.get('shelf_life_text', '-')}", (660, y), 0.5, (210, 230, 245), 1)
        y += 30
        put_text(canvas, "营养成分", (660, y), 0.58, (255, 255, 255), 1)
        y += 28
        nutrition = [
            f"能量 {record.get('calories', '-')}",
            f"蛋白质 {record.get('protein', '-')}",
            f"脂肪 {record.get('fat', '-')}",
            f"碳水 {record.get('carbohydrate', '-')}",
            f"钠 {record.get('sodium', '-')}",
        ]
        for line in nutrition:
            put_text(canvas, line, (660, y), 0.47, (190, 215, 235), 1)
            y += 24
        for line in wrap_cn_text(record.get("advice", ""), 16)[:3]:
            put_text(canvas, line, (660, y + 4), 0.45, (120, 220, 170), 1)
            y += 24
    else:
        put_text(canvas, "尚未录入标签。", (660, y), 0.56, (180, 205, 225), 1)
        put_text(canvas, "按 C 抓拍，按 S 保存解析。", (660, y + 32), 0.5, (150, 175, 195), 1)

    draw_button(canvas, 660, 508, 735, 543, "抓拍", "label_capture", False)
    draw_button(canvas, 750, 508, 825, 543, "保存", "label_save", False)
    draw_button(canvas, 840, 508, 915, 543, "清空", "label_clear", False)

    draw_status_bar(canvas, fps, infer_ms, stable_votes, frame_quality)
    return canvas


def draw_assistant_page(db, assistant, cloud_analysis=None, fps=0, infer_ms=0, stable_votes=0, frame_quality=None):
    canvas = np.full((UI_H, UI_W, 3), (12, 17, 24), dtype=np.uint8)
    draw_header(canvas, "智能语音助手", "库存问答、食用建议、提醒设置")

    draw_round_rect(canvas, 32, 90, 382, 430, (25, 32, 43), -1, 16)
    put_text(canvas, "当前库存", (55, 125), 0.72, (255, 255, 255), 2)

    y = 158
    items = db.get_inventory_items()
    has_stock = False
    for item in items:
        if item["count"] <= 0:
            continue
        has_stock = True
        color = CLASS_COLOR.get(item["class_name"], (90, 170, 255))
        cv2.circle(canvas, (60, y - 6), 8, color, -1)
        put_text(
            canvas,
            f"{item['display_name']} {format_food_count(item['class_name'], item['count'])}  建议{item['shelf_life_days']}天内",
            (78, y),
            0.5,
            (220, 235, 245),
            1,
        )
        if item.get("latest_label"):
            put_text(
                canvas,
                build_label_summary(item["latest_label"])[:20],
                (78, y + 22),
                0.42,
                (120, 220, 170),
                1,
            )
            y += 56
        else:
            y += 38

    if not has_stock:
        put_text(canvas, "当前库存为空。", (55, 170), 0.6, (170, 190, 210), 1)

    draw_round_rect(canvas, 32, 448, 382, 548, (25, 32, 43), -1, 16)
    put_text(canvas, "快捷操作", (55, 475), 0.58, (255, 255, 255), 2)
    draw_button(canvas, 55, 488, 145, 516, "查库存", "ask_inventory", False)
    draw_button(canvas, 160, 488, 250, 516, "建议", "ask_advice", False)
    draw_button(canvas, 265, 488, 355, 516, "提醒", "set_reminder", False)
    draw_button(canvas, 55, 524, 145, 548, "云分析", "cloud_analyze", False)
    draw_button(canvas, 160, 524, 250, 548, "播报建议", "speak_cloud_advice", False)
    draw_button(canvas, 265, 524, 355, 548, "语音提问", "voice_question", False)

    draw_round_rect(canvas, 410, 90, 970, 350, (25, 32, 43), -1, 16)
    put_text(canvas, "助手回答", (438, 125), 0.68, (255, 255, 255), 2)

    if assistant.latest_question:
        put_text(canvas, f"问题：{assistant.latest_question}", (438, 160), 0.5, (155, 180, 205), 1)

    answer_y = 198
    for line in wrap_cn_text(assistant.latest_answer, 28)[:7]:
        put_text(canvas, line, (438, answer_y), 0.56, (225, 238, 248), 1)
        answer_y += 34

    draw_round_rect(canvas, 410, 372, 970, 548, (25, 32, 43), -1, 16)
    put_text(canvas, "云端智能分析", (438, 410), 0.68, (255, 255, 255), 2)
    if cloud_analysis is not None:
        cloud_state = cloud_analysis.get_state()
    else:
        cloud_state = {
            "status_text": "等待云端分析",
            "source": "",
            "answer": "暂无云端建议。",
            "voice_status": "idle",
            "voice_question": "",
            "voice_answer": "",
        }

    llm_status = "DeepSeek已连接" if assistant.llm_client.available() else "本地回退"
    put_text(
        canvas,
        f"{llm_status} | {cloud_state.get('status_text', '')}",
        (438, 442),
        0.5,
        (155, 190, 230),
        1,
    )

    yy = 472
    answer = cloud_state.get("answer") or "暂无云端建议。"
    for line in wrap_cn_text(answer, 30)[:3]:
        put_text(canvas, line, (438, yy), 0.48, (225, 238, 248), 1)
        yy += 25

    voice_question = cloud_state.get("voice_question") or ""
    voice_answer = cloud_state.get("voice_answer") or ""
    if voice_question:
        put_text(canvas, f"语音：{voice_question[:22]}", (438, 528), 0.42, (170, 220, 180), 1)
    if voice_answer:
        put_text(canvas, f"回答：{voice_answer[:24]}", (680, 528), 0.42, (170, 220, 180), 1)

    draw_status_bar(canvas, fps, infer_ms, stable_votes, frame_quality)
    return canvas


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for btn in UI_STATE["buttons"]:
        x1, y1, x2, y2 = btn["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            action = btn["action"]

            if action in [PAGE_DETECT, PAGE_INVENTORY, PAGE_LABEL, PAGE_EVENTS, PAGE_ASSISTANT]:
                UI_STATE["page"] = action
            else:
                UI_STATE["pending_action"] = action

            print("[UI] click:", action)
            return


# ============================================================
# Main
# ============================================================

def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()

    runtime_config = load_runtime_config(config_args.config)
    apply_runtime_config(runtime_config)

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--model", default=choose_default_model())
    parser.add_argument("--classes", default=CLASSES_PATH)
    parser.add_argument("--camera", default=CAMERA_DEVICE, help="camera device path, or 'auto' to scan /dev/video*")
    parser.add_argument("--conf", type=float, default=CONF_THRES)
    parser.add_argument("--interval", type=int, default=INFER_INTERVAL)
    parser.add_argument("--mode", choices=["RGB", "BGR"], default="RGB")
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--init-current", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--tts-command", default=os.getenv("FRIDGE_TTS_COMMAND", TTS_COMMAND))
    parser.add_argument("--llm-api-url", default=os.getenv("FRIDGE_LLM_API_URL", LLM_API_URL))
    parser.add_argument("--llm-api-key", default=os.getenv("FRIDGE_LLM_API_KEY", LLM_API_KEY))
    parser.add_argument("--llm-model", default=os.getenv("FRIDGE_LLM_MODEL", LLM_MODEL))
    parser.add_argument("--record", default=RECORD_PATH)
    parser.add_argument("--record-fps", type=float, default=RECORD_FPS)
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
    print("[INFO] interval:", args.interval)
    print("[INFO] mode:", args.mode)
    print("[INFO] config:", args.config)
    print("[INFO] db:", DB_PATH)
    speech_enabled = ENABLE_SPEECH and not args.no_speech
    print("[INFO] speech:", speech_enabled)
    print("[INFO] audio device:", AUDIO_DEVICE)
    print("[INFO] tts command:", "custom" if args.tts_command else "espeak-ng zh")
    print("[INFO] wav prompts:", len(SPEECH_WAV_MAP), SPEECH_WAV_DIR)
    print("[INFO] llm:", "enabled" if args.llm_api_url and args.llm_model else "local fallback")
    print("[INFO] record:", args.record if args.record else "disabled")
    print("[INFO] preprocess:", PREPROCESS_METHOD if PREPROCESS_ENABLED else "disabled")

    db = InventoryDB(DB_PATH)
    speaker = Speaker(
        enable=speech_enabled,
        audio_device=AUDIO_DEVICE,
        tts_command=args.tts_command,
        wav_dir=SPEECH_WAV_DIR,
        wav_map=SPEECH_WAV_MAP,
    )
    llm_client = CloudLLMClient(
        api_url=args.llm_api_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
    )
    assistant = FridgeAssistant(db=db, speaker=speaker, llm_client=llm_client)
    cloud_analysis = CloudAnalysisManager(db=db, speaker=speaker, llm_client=llm_client)
    recorder = UIRecorder(args.record, fps=args.record_fps)
    stable_counter = StableCounter(STABLE_WINDOW, STABLE_VOTES)
    last_event_time = {name: 0.0 for name in CLASS_ORDER}

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
        db.close()
        return

    cap = open_camera(args.camera)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    speaker.say("系统就绪")

    frame_id = 0
    fps = 0.0
    infer_ms = 0.0
    last_time = time.time()

    last_detections = []
    last_detect_counts = empty_counts()
    current_stable_counts = None
    current_stable_votes = 0
    frame_quality = None

    init_current_done = False

    try:
        while True:
            ret, frame_bgr = cap.read()

            if not ret or frame_bgr is None:
                print("[WARN] failed to read frame")
                time.sleep(0.02)
                continue

            frame_id += 1

            now = time.time()
            dt = now - last_time
            last_time = now

            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            orig_h, orig_w = frame_bgr.shape[:2]
            frame_quality = assess_frame_quality(frame_bgr)

            if frame_id % args.interval == 0:
                if frame_quality.get("freeze"):
                    last_detections = []
                    last_detect_counts = empty_counts()
                    current_stable_votes = 0
                    if frame_id % max(1, args.interval * 20) == 0:
                        print("[QUALITY] freeze inventory update:", frame_quality)
                elif args.mode == "RGB":
                    infer_frame_bgr = preprocess_frame_for_inference(frame_bgr)
                    input_src = cv2.cvtColor(infer_frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    input_src = preprocess_frame_for_inference(frame_bgr)

                if not frame_quality.get("freeze"):
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
                            print("[WARN] obj and cls are all zero, check RKNN model")
                            last_detections = []
                            last_detect_counts = empty_counts()
                        else:
                            raw_detections = postprocess(
                                pred=pred,
                                classes=classes,
                                conf_thres=args.conf,
                                scale=scale,
                                pad_w=pad_w,
                                pad_h=pad_h,
                                orig_w=orig_w,
                                orig_h=orig_h,
                            )
                            last_detections = filter_detections_by_content(raw_detections, frame_bgr)

                            last_detect_counts = count_detections(last_detections, frame_bgr)

                            stable_counts, votes = stable_counter.update(last_detect_counts)
                            current_stable_votes = votes

                            if stable_counts is not None:
                                current_stable_counts = stable_counts

                                if args.init_current and not init_current_done:
                                    db.set_all_counts(stable_counts)
                                    init_current_done = True
                                    print("[INIT] DB initialized from first stable snapshot:", stable_counts)
                                    speaker.say("库存已按当前画面初始化")
                                    cloud_analysis.request_analysis(reason="init_current")
                                else:
                                    apply_inventory_change(
                                        db=db,
                                        speaker=speaker,
                                        stable_counts=stable_counts,
                                        last_event_time=last_event_time,
                                        cloud_analysis=cloud_analysis,
                                    )

            # UI pending actions from mouse click
            action = UI_STATE.get("pending_action")
            if action:
                UI_STATE["pending_action"] = None

                if action == "quit":
                    print("[UI] quit requested")
                    break

                if action == "reset":
                    db.reset_zero()
                    last_event_time = {name: 0.0 for name in CLASS_ORDER}
                    clear_label_ui_state(status="库存已清空，标签信息已清空。")
                    cloud_analysis.clear("库存已清空，等待新的云分析。")
                    print("[UI] database reset")
                    speaker.say("库存已清空")

                if action == "init":
                    if current_stable_counts is not None:
                        db.set_all_counts(current_stable_counts)
                        for class_name, count in current_stable_counts.items():
                            if normalize_count_value(count) <= 0:
                                clear_label_ui_state(class_name, "当前初始化结果为空，标签信息已清空。")
                        print("[UI] database initialized:", current_stable_counts)
                        speaker.say("库存已按当前画面初始化")
                        cloud_analysis.request_analysis(reason="manual_init")
                    else:
                        print("[UI] no stable snapshot yet")
                        speaker.say("当前没有稳定识别结果")

                if action == "ask_inventory":
                    UI_STATE["page"] = PAGE_ASSISTANT
                    assistant.ask("当前冰箱里有什么？")

                if action == "ask_advice":
                    UI_STATE["page"] = PAGE_ASSISTANT
                    assistant.ask("这些食材建议几天内食用？")

                if action == "set_reminder":
                    UI_STATE["page"] = PAGE_ASSISTANT
                    assistant.ask("请为当前库存设置食用提醒")

                if action.startswith("inventory_select_"):
                    selected = action.replace("inventory_select_", "", 1)
                    if selected in CLASS_ORDER:
                        UI_STATE["page"] = PAGE_INVENTORY
                        UI_STATE["inventory_selected_class"] = selected
                        UI_STATE["inventory_detail_scroll"] = 0

                if action == "inventory_scroll_up":
                    UI_STATE["page"] = PAGE_INVENTORY
                    UI_STATE["inventory_detail_scroll"] = max(
                        0,
                        int(UI_STATE.get("inventory_detail_scroll", 0)) - 1,
                    )

                if action == "inventory_scroll_down":
                    UI_STATE["page"] = PAGE_INVENTORY
                    UI_STATE["inventory_detail_scroll"] = int(UI_STATE.get("inventory_detail_scroll", 0)) + 1

                if action == "cloud_analyze":
                    UI_STATE["page"] = PAGE_ASSISTANT
                    cloud_analysis.request_analysis(reason="manual")

                if action == "speak_cloud_advice":
                    UI_STATE["page"] = PAGE_ASSISTANT
                    cloud_analysis.speak_latest()

                if action == "voice_question":
                    UI_STATE["page"] = PAGE_ASSISTANT
                    cloud_analysis.request_voice_question(SIMULATED_VOICE_QUESTION)

                if action == "label_target_milk":
                    UI_STATE["label_target"] = "milk_box"
                    UI_STATE["label_record"] = None
                    UI_STATE["label_status"] = "已选择牛奶标签。"

                if action == "label_target_coke":
                    UI_STATE["label_target"] = "coke_can"
                    UI_STATE["label_record"] = None
                    UI_STATE["label_status"] = "已选择可乐标签。"

                if action == "label_capture":
                    capture_path = save_label_frame(frame_bgr)
                    if capture_path:
                        UI_STATE["label_capture_path"] = capture_path
                        UI_STATE["label_status"] = "抓拍完成，请保存解析。"
                    else:
                        UI_STATE["label_status"] = "抓拍失败，请重新对准。"
                        speaker.say(LABEL_FAILED_TEXT, prompt_key="label_failed")

                if action == "label_save":
                    ok, record = save_label_capture_record(
                        db,
                        speaker,
                        UI_STATE.get("label_target", "milk_box"),
                        UI_STATE.get("label_capture_path", ""),
                    )
                    if ok:
                        UI_STATE["label_record"] = record
                        UI_STATE["label_status"] = "标签录入成功。"
                    else:
                        UI_STATE["label_status"] = "标签识别失败。"

                if action == "label_clear":
                    clear_label_ui_state(status="已清空标签结果。")

            db_counts = db.get_counts()

            if UI_STATE["page"] == PAGE_DETECT:
                canvas = draw_detect_page(
                    frame=frame_bgr,
                    detections=last_detections,
                    detect_counts=last_detect_counts,
                    stable_counts=current_stable_counts,
                    db_counts=db_counts,
                    fps=fps,
                    infer_ms=infer_ms,
                    stable_votes=current_stable_votes,
                    frame_quality=frame_quality,
                )
            elif UI_STATE["page"] == PAGE_INVENTORY:
                canvas = draw_inventory_page(
                    db=db,
                    detect_counts=last_detect_counts,
                    stable_counts=current_stable_counts,
                    db_counts=db_counts,
                    fps=fps,
                    infer_ms=infer_ms,
                    stable_votes=current_stable_votes,
                    frame_quality=frame_quality,
                )
            elif UI_STATE["page"] == PAGE_LABEL:
                canvas = draw_label_page(
                    frame=frame_bgr,
                    db=db,
                    fps=fps,
                    infer_ms=infer_ms,
                    stable_votes=current_stable_votes,
                    frame_quality=frame_quality,
                )
            elif UI_STATE["page"] == PAGE_EVENTS:
                canvas = draw_events_page(
                    db=db,
                    fps=fps,
                    infer_ms=infer_ms,
                    stable_votes=current_stable_votes,
                    frame_quality=frame_quality,
                )
            else:
                canvas = draw_assistant_page(
                    db=db,
                    assistant=assistant,
                    cloud_analysis=cloud_analysis,
                    fps=fps,
                    infer_ms=infer_ms,
                    stable_votes=current_stable_votes,
                    frame_quality=frame_quality,
                )

            cv2.imshow(WINDOW_NAME, canvas)
            recorder.write(canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("1"):
                UI_STATE["page"] = PAGE_DETECT

            if key == ord("2"):
                UI_STATE["page"] = PAGE_INVENTORY

            if key == ord("3"):
                UI_STATE["page"] = PAGE_EVENTS

            if key == ord("4"):
                UI_STATE["page"] = PAGE_ASSISTANT

            if key == ord("5") or key == ord("l"):
                UI_STATE["page"] = PAGE_LABEL

            if key == ord("c"):
                UI_STATE["pending_action"] = "label_capture"

            if key == ord("s"):
                UI_STATE["pending_action"] = "label_save"

            if key == ord("x"):
                UI_STATE["pending_action"] = "label_clear"

            if key == ord("a"):
                UI_STATE["page"] = PAGE_ASSISTANT
                assistant.ask("当前冰箱里有什么？")

            if key == ord("d"):
                UI_STATE["page"] = PAGE_ASSISTANT
                assistant.ask("这些食材建议几天内食用？")

            if key == ord("r"):
                UI_STATE["page"] = PAGE_ASSISTANT
                assistant.ask("请为当前库存设置食用提醒")

            if key == ord("z"):
                db.reset_zero()
                last_event_time = {name: 0.0 for name in CLASS_ORDER}
                clear_label_ui_state(status="库存已清空，标签信息已清空。")
                cloud_analysis.clear("库存已清空，等待新的云分析。")
                print("[MANUAL] database reset to zero")
                speaker.say("库存已清空")

            if key == ord("i"):
                if current_stable_counts is not None:
                    db.set_all_counts(current_stable_counts)
                    for class_name, count in current_stable_counts.items():
                        if normalize_count_value(count) <= 0:
                            clear_label_ui_state(class_name, "当前初始化结果为空，标签信息已清空。")
                    print("[MANUAL] database initialized:", current_stable_counts)
                    speaker.say("库存已按当前画面初始化")
                    cloud_analysis.request_analysis(reason="manual_init")
                else:
                    print("[MANUAL] no stable snapshot yet")
                    speaker.say("当前没有稳定识别结果")

    finally:
        cap.release()
        rknn.release()
        recorder.close()
        db.close()
        cv2.destroyAllWindows()
        print("[INFO] UI demo stopped")


if __name__ == "__main__":
    main()
