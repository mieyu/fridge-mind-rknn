import importlib.util
import json
import sqlite3
import sys
import tempfile
import types
import wave
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("camera_inventory_ui_demo.py")


def load_demo_module():
    try:
        import cv2  # noqa: F401
    except Exception:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.FONT_HERSHEY_SIMPLEX = 0
        cv2_stub.LINE_AA = 16
        cv2_stub.WINDOW_NORMAL = 0
        cv2_stub.WND_PROP_FULLSCREEN = 0
        cv2_stub.WINDOW_FULLSCREEN = 1
        cv2_stub.EVENT_LBUTTONDOWN = 1
        cv2_stub.CAP_PROP_FOURCC = 6
        cv2_stub.CAP_PROP_FRAME_WIDTH = 3
        cv2_stub.CAP_PROP_FRAME_HEIGHT = 4
        cv2_stub.CAP_PROP_FPS = 5
        cv2_stub.CAP_PROP_BUFFERSIZE = 38
        cv2_stub.COLOR_BGR2RGB = 4
        cv2_stub.COLOR_BGR2HSV = 40
        cv2_stub.COLOR_BGR2GRAY = 6
        cv2_stub.INTER_LINEAR = 1
        cv2_stub.dnn = types.SimpleNamespace(NMSBoxes=lambda **kwargs: [])
        cv2_stub.rectangle = lambda *args, **kwargs: None
        cv2_stub.circle = lambda *args, **kwargs: None
        cv2_stub.putText = lambda *args, **kwargs: None
        cv2_stub.getTextSize = lambda text, font, scale, thickness: ((len(str(text)) * 10, 20), 0)
        cv2_stub.VideoWriter_fourcc = lambda *args: 0
        sys.modules.setdefault("cv2", cv2_stub)

    rknn_api_stub = types.ModuleType("rknnlite.api")

    class RKNNLite:
        pass

    rknn_api_stub.RKNNLite = RKNNLite
    sys.modules.setdefault("rknnlite", types.ModuleType("rknnlite"))
    sys.modules["rknnlite.api"] = rknn_api_stub

    spec = importlib.util.spec_from_file_location("camera_inventory_ui_demo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chinese_speech_and_inventory_answer():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("apple", 2)
            db.set_count("milk_box", 1)

            speak_text = demo.build_speak_text("apple", 1, 2)
            assert speak_text == "苹果增加1个，当前2个"

            answer = demo.build_inventory_answer(db)
            assert "苹果2个" in answer
            assert "盒装牛奶1盒" in answer
            assert "建议苹果7天内食用" in answer

            reminder_text = demo.build_reminder_answer(db)
            assert "已设置2条食用提醒" in reminder_text

            reminders = db.get_reminders(limit=10)
            assert len(reminders) == 2
            assert reminders[0][1] in {"apple", "milk_box"}
        finally:
            db.close()


def test_measure_text_supports_old_pillow_font_api():
    demo = load_demo_module()

    class OldPillowFont:
        def getsize(self, text):
            return (len(text) * 12, 18)

    original_get_pil_font = demo.get_pil_font
    try:
        demo.get_pil_font = lambda scale: OldPillowFont()
        assert demo.measure_text("苹果", 0.55, 1) == (24, 18)
    finally:
        demo.get_pil_font = original_get_pil_font


def test_chinese_text_sprite_is_cached():
    demo = load_demo_module()
    if demo.Image is None:
        return

    demo.TEXT_RENDER_CACHE.clear()
    first = demo.render_text_sprite("识别", 0.55, (255, 255, 255))
    second = demo.render_text_sprite("识别", 0.55, (255, 255, 255))

    assert first is second
    assert len(demo.TEXT_RENDER_CACHE) == 1


def test_put_text_uses_baseline_for_chinese_sprite():
    demo = load_demo_module()
    np = __import__("numpy")
    canvas = np.zeros((40, 80, 3), dtype=np.uint8)
    sprite = (
        np.zeros((10, 20, 3), dtype=np.uint8),
        np.full((10, 20), 255, dtype=np.uint8),
    )
    calls = []

    original_render = demo.render_text_sprite
    original_blit = demo.blit_text_sprite
    try:
        demo.render_text_sprite = lambda text, scale, color: sprite
        demo.blit_text_sprite = lambda img, sp, pos: calls.append(pos)
        demo.put_text(canvas, "识别", (5, 30), 0.55, (255, 255, 255), 1)
    finally:
        demo.render_text_sprite = original_render
        demo.blit_text_sprite = original_blit

    assert calls == [(5, 20)]


def test_header_registers_quit_button():
    demo = load_demo_module()
    np = __import__("numpy")
    canvas = np.zeros((demo.UI_H, demo.UI_W, 3), dtype=np.uint8)

    demo.draw_header(canvas, "标题", "副标题")
    actions = [button["action"] for button in demo.UI_STATE["buttons"]]

    assert "quit" in actions


def test_header_registers_label_page_button():
    demo = load_demo_module()
    np = __import__("numpy")
    canvas = np.zeros((demo.UI_H, demo.UI_W, 3), dtype=np.uint8)

    demo.draw_header(canvas, "标题", "副标题")
    actions = [button["action"] for button in demo.UI_STATE["buttons"]]

    assert demo.PAGE_LABEL in actions


def test_ui_recorder_writes_frames():
    demo = load_demo_module()
    np = __import__("numpy")
    created = []

    class FakeWriter:
        def __init__(self, *args):
            self.args = args
            self.frames = []
            created.append(self)

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    original_writer = demo.cv2.VideoWriter
    try:
        demo.cv2.VideoWriter = FakeWriter
        recorder = demo.UIRecorder("demo.avi", fps=20)
        recorder.write(np.zeros((12, 20, 3), dtype=np.uint8))
        recorder.close()
    finally:
        demo.cv2.VideoWriter = original_writer

    assert len(created) == 1
    assert len(created[0].frames) == 1
    assert created[0].frames[0].shape == (12, 20, 3)


def test_speaker_prefers_wav_prompt_when_available():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "system_ready.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"system_ready": "system_ready.wav"},
        )
        cmd = speaker.build_speech_command("系统就绪")

    assert cmd.startswith("python3 ")
    assert "normalize-audio" in cmd
    assert "system_ready.wav" in cmd
    assert "espeak-ng" not in cmd


def test_normalize_wav_file_converts_mono_44k_to_stereo_48k():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / "source.wav"
        dst = Path(tmp_dir) / "normalized.wav"
        with wave.open(str(src), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x00\x00" * 4410)

        demo.normalize_wav_file(src, dst)

        with wave.open(str(dst), "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 48000
            assert wf.getnframes() > 0


def test_speaker_falls_back_to_tts_without_wav_prompt():
    demo = load_demo_module()
    speaker = demo.Speaker(
        enable=False,
        audio_device="plughw:1,0",
        wav_dir="",
        wav_map={},
    )
    cmd = speaker.build_speech_command("动态回答内容")

    assert "espeak-ng -v zh" in cmd
    assert " -w " in cmd
    assert "normalize-audio" in cmd
    assert "aplay -D plughw:1,0" in cmd
    assert "fridge_tts_cache" in cmd


def test_speaker_builds_tts_fallback_after_wav_prompt():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "system_ready.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"system_ready": "system_ready.wav"},
        )
        commands = speaker.build_speech_commands("系统就绪")

    assert len(commands) == 2
    assert commands[0][0] == "wav"
    assert "normalize-audio" in commands[0][1]
    assert "system_ready.wav" in commands[0][1]
    assert commands[1][0] == "tts"
    assert "espeak-ng -v zh" in commands[1][1]
    assert "normalize-audio" in commands[1][1]


def test_speaker_uses_explicit_prompt_key_for_wav_prompt():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "label_saved.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"label_saved": "label_saved.wav"},
        )
        commands = speaker.build_speech_commands("录入成功", prompt_key="label_saved")

    assert commands[0][0] == "wav"
    assert "label_saved.wav" in commands[0][1]


def test_speaker_retries_wav_once_before_tts_fallback():
    demo = load_demo_module()

    class Result:
        def __init__(self, returncode, stderr=""):
            self.returncode = returncode
            self.stderr = stderr

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "system_ready.wav").write_bytes(b"RIFF")
        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"system_ready": "system_ready.wav"},
        )

        calls = []
        original_run = demo.subprocess.run
        try:
            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if len(calls) == 1:
                    return Result(1, "aplay failed")
                return Result(0, "")

            demo.subprocess.run = fake_run
            assert speaker.play_text("系统就绪") is True
        finally:
            demo.subprocess.run = original_run

    assert len(calls) == 2
    assert "system_ready.wav" in calls[0]
    assert "system_ready.wav" in calls[1]


def test_speaker_falls_back_to_tts_after_wav_retries_fail():
    demo = load_demo_module()

    class Result:
        def __init__(self, returncode, stderr=""):
            self.returncode = returncode
            self.stderr = stderr

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "system_ready.wav").write_bytes(b"RIFF")
        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"system_ready": "system_ready.wav"},
        )

        calls = []
        original_run = demo.subprocess.run
        try:
            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if "system_ready.wav" in cmd:
                    return Result(1, "aplay failed")
                return Result(0, "")

            demo.subprocess.run = fake_run
            assert speaker.play_text("系统就绪") is True
        finally:
            demo.subprocess.run = original_run

    assert len(calls) == 3
    assert "system_ready.wav" in calls[0]
    assert "system_ready.wav" in calls[1]
    assert "espeak-ng -v zh" in calls[2]


def test_speaker_times_out_stuck_commands_and_continues_queue():
    demo = load_demo_module()

    class Result:
        def __init__(self, returncode, stderr=""):
            self.returncode = returncode
            self.stderr = stderr

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "system_ready.wav").write_bytes(b"RIFF")
        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"system_ready": "system_ready.wav"},
            command_retries=1,
            command_timeout=0.01,
        )

        calls = []
        original_run = demo.subprocess.run
        try:
            def fake_run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                if len(calls) == 1:
                    raise demo.subprocess.TimeoutExpired(cmd, timeout=kwargs.get("timeout"))
                return Result(0, "")

            demo.subprocess.run = fake_run
            assert speaker.play_text("系统就绪") is True
        finally:
            demo.subprocess.run = original_run

        assert len(calls) == 2
        assert calls[0][1]["timeout"] == 0.01
        assert calls[1][1]["timeout"] == 0.01
        assert "system_ready.wav" in calls[0][0]
        assert "espeak-ng -v zh" in calls[1][0]


def test_speaker_builds_hdmi_audio_init_command_for_plughw_device():
    demo = load_demo_module()

    speaker = demo.Speaker(
        enable=False,
        audio_device="plughw:1,0",
        wav_dir="",
        wav_map={},
    )

    assert speaker.build_audio_init_commands() == ["amixer -c 1 sset 'ELD Bypass' on"]


def test_speaker_uses_event_wav_for_inventory_addition():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "apple_add.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"apple_add": "apple_add.wav"},
        )
        cmd = speaker.build_speech_command(demo.build_speak_text("apple", 1, 1))

    assert cmd.startswith("python3 ")
    assert "normalize-audio" in cmd
    assert "apple_add.wav" in cmd
    assert "espeak-ng" not in cmd


def test_speaker_uses_event_wav_for_inventory_removal():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "milk_remove.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"milk_remove": "milk_remove.wav"},
        )
        cmd = speaker.build_speech_command(demo.build_speak_text("milk_box", -1, 0))

    assert cmd.startswith("python3 ")
    assert "normalize-audio" in cmd
    assert "milk_remove.wav" in cmd
    assert "espeak-ng" not in cmd


def test_speaker_uses_blueberry_half_wav_for_half_box_change():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "blueberry_half.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"blueberry_half": "blueberry_half.wav"},
        )
        cmd = speaker.build_speech_command(demo.build_speak_text("blueberry_box", -0.5, 0.5))

    assert cmd.startswith("python3 ")
    assert "normalize-audio" in cmd
    assert "blueberry_half.wav" in cmd
    assert "espeak-ng" not in cmd


def test_speaker_uses_inventory_empty_wav_for_empty_answer():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "inventory_empty.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"inventory_empty": "inventory_empty.wav"},
        )
        cmd = speaker.build_speech_command("当前冰箱库存为空。")

    assert cmd.startswith("python3 ")
    assert "normalize-audio" in cmd
    assert "inventory_empty.wav" in cmd
    assert "espeak-ng" not in cmd


def test_speaker_uses_reminder_saved_wav_for_dynamic_reminder_answer():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_dir = Path(tmp_dir)
        (wav_dir / "reminder_saved.wav").write_bytes(b"RIFF")

        speaker = demo.Speaker(
            enable=False,
            audio_device="plughw:1,0",
            wav_dir=str(wav_dir),
            wav_map={"reminder_saved": "reminder_saved.wav"},
        )
        cmd = speaker.build_speech_command("已设置2条食用提醒，请按建议时间优先处理临期食材。")

    assert cmd.startswith("python3 ")
    assert "normalize-audio" in cmd
    assert "reminder_saved.wav" in cmd
    assert "espeak-ng" not in cmd


def test_estimates_blueberry_half_box_from_blue_area():
    demo = load_demo_module()
    np = __import__("numpy")
    original_method = demo.BLUEBERRY_AREA_METHOD
    original_shrink = demo.BLUEBERRY_ROI_SHRINK
    try:
        demo.BLUEBERRY_AREA_METHOD = "hsv"
        demo.BLUEBERRY_ROI_SHRINK = 0.0
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        frame[10:70, 10:70] = (30, 30, 30)
        frame[20:65, 20:65] = (160, 40, 20)
        assert demo.estimate_blueberry_box_count(frame, [10, 10, 70, 70]) == 1.0

        frame[20:65, 20:65] = (30, 30, 30)
        frame[30:52, 20:60] = (160, 40, 20)
        assert demo.estimate_blueberry_box_count(frame, [10, 10, 70, 70]) == 0.5
    finally:
        demo.BLUEBERRY_AREA_METHOD = original_method
        demo.BLUEBERRY_ROI_SHRINK = original_shrink


def test_estimates_blueberry_full_box_from_binary_dark_area():
    demo = load_demo_module()
    np = __import__("numpy")

    original = {
        "BLUEBERRY_AREA_METHOD": demo.BLUEBERRY_AREA_METHOD,
        "BLUEBERRY_BINARY_THRESHOLD": demo.BLUEBERRY_BINARY_THRESHOLD,
        "BLUEBERRY_ROI_SHRINK": demo.BLUEBERRY_ROI_SHRINK,
        "BLUEBERRY_FULL_RATIO": demo.BLUEBERRY_FULL_RATIO,
        "BLUEBERRY_HALF_RATIO": demo.BLUEBERRY_HALF_RATIO,
        "BLUEBERRY_COLOR_GUARD_RATIO": demo.BLUEBERRY_COLOR_GUARD_RATIO,
    }
    try:
        demo.BLUEBERRY_AREA_METHOD = "binary_dark"
        demo.BLUEBERRY_BINARY_THRESHOLD = 110
        demo.BLUEBERRY_ROI_SHRINK = 0.0
        demo.BLUEBERRY_FULL_RATIO = 0.45
        demo.BLUEBERRY_HALF_RATIO = 0.12
        demo.BLUEBERRY_COLOR_GUARD_RATIO = 0.0

        frame = np.full((80, 80, 3), 190, dtype=np.uint8)
        frame[12:68, 12:68] = 180
        frame[18:66, 18:66] = 55
        assert demo.estimate_blueberry_box_count(frame, [10, 10, 70, 70]) == 1.0

        frame[18:66, 18:66] = 180
        frame[34:54, 20:62] = 55
        assert demo.estimate_blueberry_box_count(frame, [10, 10, 70, 70]) == 0.5
    finally:
        for key, value in original.items():
            setattr(demo, key, value)


def test_rejects_blueberry_shadow_without_blue_purple_content():
    demo = load_demo_module()
    np = __import__("numpy")

    original = {
        "BLUEBERRY_AREA_METHOD": demo.BLUEBERRY_AREA_METHOD,
        "BLUEBERRY_BINARY_THRESHOLD": demo.BLUEBERRY_BINARY_THRESHOLD,
        "BLUEBERRY_ROI_SHRINK": demo.BLUEBERRY_ROI_SHRINK,
        "BLUEBERRY_FULL_RATIO": demo.BLUEBERRY_FULL_RATIO,
        "BLUEBERRY_HALF_RATIO": demo.BLUEBERRY_HALF_RATIO,
        "BLUEBERRY_COLOR_GUARD_RATIO": demo.BLUEBERRY_COLOR_GUARD_RATIO,
    }
    try:
        demo.BLUEBERRY_AREA_METHOD = "binary_dark"
        demo.BLUEBERRY_BINARY_THRESHOLD = 110
        demo.BLUEBERRY_ROI_SHRINK = 0.0
        demo.BLUEBERRY_FULL_RATIO = 0.45
        demo.BLUEBERRY_HALF_RATIO = 0.12
        demo.BLUEBERRY_COLOR_GUARD_RATIO = 0.02

        frame = np.full((80, 80, 3), 190, dtype=np.uint8)
        frame[18:66, 18:66] = 45
        box = [10, 10, 70, 70]

        assert demo.estimate_blueberry_box_count(frame, box) == 0.0
        detections = [{"class_name": "blueberry_box", "score": 0.8, "box": box}]
        assert demo.filter_detections_by_content(detections, frame) == []
    finally:
        for key, value in original.items():
            setattr(demo, key, value)


def test_postprocess_drops_ambiguous_milk_coke_classification():
    demo = load_demo_module()
    np = __import__("numpy")

    original_margin = demo.MILK_COKE_MIN_MARGIN
    try:
        demo.MILK_COKE_MIN_MARGIN = 0.12
        pred = np.array([
            [208, 208, 100, 100, 1.0, 0.01, 0.01, 0.70, 0.74, 0.01, 0.01],
        ], dtype=np.float32)

        results = demo.postprocess(
            pred=pred,
            classes=demo.CLASS_ORDER,
            conf_thres=0.25,
            scale=1.0,
            pad_w=0,
            pad_h=0,
            orig_w=416,
            orig_h=416,
        )

        assert results == []
    finally:
        demo.MILK_COKE_MIN_MARGIN = original_margin


def test_dark_frame_quality_freezes_inventory_update():
    demo = load_demo_module()
    np = __import__("numpy")
    dark = np.zeros((80, 80, 3), dtype=np.uint8)
    normal = np.full((80, 80, 3), 120, dtype=np.uint8)
    normal[::2, :, :] = 200

    assert demo.assess_frame_quality(dark)["freeze"] is True
    assert demo.assess_frame_quality(normal)["freeze"] is False


def test_open_camera_falls_back_to_available_device():
    demo = load_demo_module()
    opened = []

    class FakeCapture:
        def __init__(self, device, backend=None):
            self.device = device
            self.backend = backend
            self.props = {}
            opened.append(device)

        def isOpened(self):
            return self.device == "/dev/video0"

        def set(self, prop, value):
            self.props[prop] = value

        def get(self, prop):
            if prop == demo.cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            if prop == demo.cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            if prop == demo.cv2.CAP_PROP_FPS:
                return 30
            return 0

        def release(self):
            self.released = True

    original_capture = demo.cv2.VideoCapture
    original_list = demo.list_video_devices
    try:
        demo.cv2.VideoCapture = FakeCapture
        demo.list_video_devices = lambda: ["/dev/video0"]
        cap = demo.open_camera("/dev/video18")
    finally:
        demo.cv2.VideoCapture = original_capture
        demo.list_video_devices = original_list

    assert cap.device == "/dev/video0"
    assert opened == ["/dev/video18", "/dev/video0"]


def test_runtime_config_overrides_tunable_parameters():
    demo = load_demo_module()
    original = {
        "BLUEBERRY_FULL_RATIO": demo.BLUEBERRY_FULL_RATIO,
        "BLUEBERRY_HALF_RATIO": demo.BLUEBERRY_HALF_RATIO,
        "BLUEBERRY_AREA_METHOD": demo.BLUEBERRY_AREA_METHOD,
        "BLUEBERRY_BINARY_THRESHOLD": demo.BLUEBERRY_BINARY_THRESHOLD,
        "BLUEBERRY_ROI_SHRINK": demo.BLUEBERRY_ROI_SHRINK,
        "BLUEBERRY_COLOR_GUARD_RATIO": demo.BLUEBERRY_COLOR_GUARD_RATIO,
        "BLUEBERRY_HSV_LOWER": demo.BLUEBERRY_HSV_LOWER.copy(),
        "BLUEBERRY_HSV_UPPER": demo.BLUEBERRY_HSV_UPPER.copy(),
        "LOW_LIGHT_MEAN": demo.LOW_LIGHT_MEAN,
        "LOW_CONTRAST_STD": demo.LOW_CONTRAST_STD,
        "BLUR_VAR_THRES": demo.BLUR_VAR_THRES,
        "STABLE_WINDOW": demo.STABLE_WINDOW,
        "STABLE_VOTES": demo.STABLE_VOTES,
        "EVENT_COOLDOWN": demo.EVENT_COOLDOWN,
        "CAMERA_DEVICE": demo.CAMERA_DEVICE,
        "INFER_INTERVAL": demo.INFER_INTERVAL,
        "CONF_THRES": demo.CONF_THRES,
        "MILK_COKE_MIN_MARGIN": demo.MILK_COKE_MIN_MARGIN,
        "AUDIO_DEVICE": demo.AUDIO_DEVICE,
        "ENABLE_SPEECH": demo.ENABLE_SPEECH,
        "PREPROCESS_ENABLED": demo.PREPROCESS_ENABLED,
        "PREPROCESS_METHOD": demo.PREPROCESS_METHOD,
        "PREPROCESS_CLAHE_CLIP_LIMIT": demo.PREPROCESS_CLAHE_CLIP_LIMIT,
        "PREPROCESS_GAMMA": demo.PREPROCESS_GAMMA,
        "PREPROCESS_SATURATION_GAIN": demo.PREPROCESS_SATURATION_GAIN,
        "PREPROCESS_SHARPEN_AMOUNT": demo.PREPROCESS_SHARPEN_AMOUNT,
        "PREPROCESS_GLARE_SUPPRESS": demo.PREPROCESS_GLARE_SUPPRESS,
        "PREPROCESS_GLARE_V_THRESHOLD": demo.PREPROCESS_GLARE_V_THRESHOLD,
        "PREPROCESS_GLARE_S_THRESHOLD": demo.PREPROCESS_GLARE_S_THRESHOLD,
    }

    config = {
        "blueberry": {
            "method": "binary_dark",
            "full_ratio": 0.18,
            "half_ratio": 0.04,
            "binary_threshold": 105,
            "roi_shrink": 0.1,
            "color_guard_ratio": 0.03,
            "hsv_lower": [88, 25, 18],
            "hsv_upper": [170, 255, 255],
        },
        "quality": {
            "low_light_mean": 35,
            "low_contrast_std": 8,
            "blur_var_threshold": 10,
        },
        "preprocess": {
            "enabled": True,
            "method": "plastic_wrap",
            "clahe_clip_limit": 1.8,
            "gamma": 0.85,
            "saturation_gain": 1.25,
            "sharpen_amount": 0.35,
            "glare_suppress": True,
            "glare_v_threshold": 230,
            "glare_s_threshold": 70,
        },
        "runtime": {
            "stable_window": 6,
            "stable_votes": 4,
            "event_cooldown": 1.2,
            "infer_interval": 9,
            "conf_threshold": 0.32,
            "milk_coke_min_margin": 0.2,
        },
        "camera": {
            "device": "/dev/video0",
        },
        "speech": {
            "enabled": False,
            "audio_device": "plughw:0,0",
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "fridge_demo_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        loaded = demo.load_runtime_config(str(config_path))

    try:
        demo.apply_runtime_config(loaded)

        assert demo.BLUEBERRY_FULL_RATIO == 0.18
        assert demo.BLUEBERRY_HALF_RATIO == 0.04
        assert demo.BLUEBERRY_AREA_METHOD == "binary_dark"
        assert demo.BLUEBERRY_BINARY_THRESHOLD == 105
        assert demo.BLUEBERRY_ROI_SHRINK == 0.1
        assert demo.BLUEBERRY_COLOR_GUARD_RATIO == 0.03
        assert demo.BLUEBERRY_HSV_LOWER.tolist() == [88, 25, 18]
        assert demo.BLUEBERRY_HSV_UPPER.tolist() == [170, 255, 255]
        assert demo.LOW_LIGHT_MEAN == 35
        assert demo.LOW_CONTRAST_STD == 8
        assert demo.BLUR_VAR_THRES == 10
        assert demo.PREPROCESS_ENABLED is True
        assert demo.PREPROCESS_METHOD == "plastic_wrap"
        assert demo.PREPROCESS_CLAHE_CLIP_LIMIT == 1.8
        assert demo.PREPROCESS_GAMMA == 0.85
        assert demo.PREPROCESS_SATURATION_GAIN == 1.25
        assert demo.PREPROCESS_SHARPEN_AMOUNT == 0.35
        assert demo.PREPROCESS_GLARE_SUPPRESS is True
        assert demo.PREPROCESS_GLARE_V_THRESHOLD == 230
        assert demo.PREPROCESS_GLARE_S_THRESHOLD == 70
        assert demo.STABLE_WINDOW == 6
        assert demo.STABLE_VOTES == 4
        assert demo.EVENT_COOLDOWN == 1.2
        assert demo.INFER_INTERVAL == 9
        assert demo.CONF_THRES == 0.32
        assert demo.MILK_COKE_MIN_MARGIN == 0.2
        assert demo.CAMERA_DEVICE == "/dev/video0"
        assert demo.AUDIO_DEVICE == "plughw:0,0"
        assert demo.ENABLE_SPEECH is False
    finally:
        for key, value in original.items():
            setattr(demo, key, value)


def test_runtime_config_resolves_relative_wav_dir_from_config_file():
    demo = load_demo_module()
    original_wav_dir = demo.SPEECH_WAV_DIR

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dir = Path(tmp_dir)
        config_path = config_dir / "fridge_demo_config.json"
        config_path.write_text(
            json.dumps({"speech": {"wav_dir": "voice_prompts"}}),
            encoding="utf-8",
        )
        loaded = demo.load_runtime_config(str(config_path))

        try:
            demo.apply_runtime_config(loaded)
            assert Path(demo.SPEECH_WAV_DIR) == config_dir / "voice_prompts"
        finally:
            demo.SPEECH_WAV_DIR = original_wav_dir


def test_project_config_uses_fixed_board_camera_and_valid_deepseek_model():
    config = json.loads(SCRIPT_PATH.with_name("fridge_demo_config.json").read_text(encoding="utf-8"))

    assert config["camera"]["device"] == "/dev/video18"
    assert config["llm"]["model"] == "deepseek-v4-flash"


def test_llm_client_logs_http_error_response_body():
    demo = load_demo_module()
    import contextlib
    import io

    body = b'{"error":{"message":"Model Not Exist"}}'
    original_urlopen = demo.urllib.request.urlopen

    def fake_urlopen(req, timeout):
        raise demo.urllib.error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(body),
        )

    output = io.StringIO()
    try:
        demo.urllib.request.urlopen = fake_urlopen
        client = demo.CloudLLMClient(
            api_url="https://api.deepseek.com/chat/completions",
            api_key="fake-key",
            model="bad-model",
        )
        with contextlib.redirect_stdout(output):
            answer = client.ask("测试问题", [])
    finally:
        demo.urllib.request.urlopen = original_urlopen

    assert answer == ""
    assert "HTTP Error 400" in output.getvalue()
    assert "Model Not Exist" in output.getvalue()


def test_preprocess_suppresses_plastic_wrap_glare():
    demo = load_demo_module()
    np = __import__("numpy")

    original = {
        "PREPROCESS_ENABLED": demo.PREPROCESS_ENABLED,
        "PREPROCESS_METHOD": demo.PREPROCESS_METHOD,
        "PREPROCESS_CLAHE_CLIP_LIMIT": demo.PREPROCESS_CLAHE_CLIP_LIMIT,
        "PREPROCESS_GAMMA": demo.PREPROCESS_GAMMA,
        "PREPROCESS_SATURATION_GAIN": demo.PREPROCESS_SATURATION_GAIN,
        "PREPROCESS_SHARPEN_AMOUNT": demo.PREPROCESS_SHARPEN_AMOUNT,
        "PREPROCESS_GLARE_SUPPRESS": demo.PREPROCESS_GLARE_SUPPRESS,
        "PREPROCESS_GLARE_V_THRESHOLD": demo.PREPROCESS_GLARE_V_THRESHOLD,
        "PREPROCESS_GLARE_S_THRESHOLD": demo.PREPROCESS_GLARE_S_THRESHOLD,
    }
    try:
        demo.PREPROCESS_ENABLED = True
        demo.PREPROCESS_METHOD = "plastic_wrap"
        demo.PREPROCESS_CLAHE_CLIP_LIMIT = 0.0
        demo.PREPROCESS_GAMMA = 1.0
        demo.PREPROCESS_SATURATION_GAIN = 1.0
        demo.PREPROCESS_SHARPEN_AMOUNT = 0.0
        demo.PREPROCESS_GLARE_SUPPRESS = True
        demo.PREPROCESS_GLARE_V_THRESHOLD = 230
        demo.PREPROCESS_GLARE_S_THRESHOLD = 80

        frame = np.full((80, 80, 3), 80, dtype=np.uint8)
        frame[20:60, 20:60] = (40, 70, 180)
        frame[34:38, 34:38] = (255, 255, 255)

        enhanced = demo.preprocess_frame_for_inference(frame)
        assert enhanced.shape == frame.shape
        assert enhanced[35:37, 35:37].mean() < frame[35:37, 35:37].mean()
    finally:
        for key, value in original.items():
            setattr(demo, key, value)


def test_parse_milk_sample_ocr_extracts_label_fields():
    demo = load_demo_module()

    record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])

    assert record["class_name"] == "milk_box"
    assert record["product_name"] == "纯牛奶"
    assert record["brand"] == "演示牧场"
    assert record["shelf_life_text"] == "6个月"
    assert record["calories"] == "280 kJ"
    assert record["protein"] == "3.2 g"
    assert record["fat"] == "3.6 g"
    assert record["carbohydrate"] == "4.8 g"
    assert record["sodium"] == "60 mg"
    assert "优先饮用" in record["advice"]


def test_parse_coke_sample_ocr_extracts_label_fields():
    demo = load_demo_module()

    record = demo.parse_label_ocr_text("coke_can", demo.LABEL_SAMPLE_OCR_TEXT["coke_can"])

    assert record["class_name"] == "coke_can"
    assert record["product_name"] == "罐装可乐"
    assert record["brand"] == "演示汽水"
    assert record["shelf_life_text"] == "12个月"
    assert record["calories"] == "180 kJ"
    assert record["protein"] == "0 g"
    assert record["fat"] == "0 g"
    assert record["carbohydrate"] == "10.6 g"
    assert record["sodium"] == "12 mg"
    assert "适量饮用" in record["advice"]


def test_label_records_are_saved_and_latest_record_is_returned():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            old_record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            old_record["product_name"] = "旧牛奶"
            old_record["image_path"] = str(Path(tmp_dir) / "old.jpg")
            db.save_label_record(old_record)

            new_record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            new_record["product_name"] = "新牛奶"
            new_record["image_path"] = str(Path(tmp_dir) / "new.jpg")
            db.save_label_record(new_record)

            latest = db.get_latest_label_record("milk_box")
            assert latest["class_name"] == "milk_box"
            assert latest["product_name"] == "新牛奶"
            assert latest["image_path"].endswith("new.jpg")
            assert latest["protein"] == "3.2 g"
        finally:
            db.close()


def test_inventory_answer_uses_latest_label_record_for_packaged_food():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)
            record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            db.save_label_record(record)

            items = db.get_inventory_items()
            milk = next(item for item in items if item["class_name"] == "milk_box")
            assert milk["latest_label"]["product_name"] == "纯牛奶"
            assert milk["latest_label"]["shelf_life_text"] == "6个月"

            answer = demo.build_inventory_answer(db)
            assert "纯牛奶" in answer
            assert "保质期6个月" in answer
            assert "蛋白质3.2 g" in answer
        finally:
            db.close()


def test_inventory_detail_for_packaged_food_requires_label_record():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)

            detail = demo.build_inventory_detail(db, "milk_box")

            assert detail["status"] == "missing_label"
            assert "尚未完成标签录入" in "\n".join(detail["lines"])
            assert detail["title"] == "盒装牛奶"
        finally:
            db.close()


def test_inventory_detail_for_packaged_food_uses_label_nutrition():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)
            record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            db.save_label_record(record)

            detail = demo.build_inventory_detail(db, "milk_box")
            joined = "\n".join(detail["lines"])

            assert detail["status"] == "ok"
            assert "产品：纯牛奶" in joined
            assert "能量：280 kJ" in joined
            assert "蛋白质：3.2 g" in joined
            assert ("品牌", "演示牧场") in detail["metadata"]
            assert ("生产日期", "2026-06-01") in detail["metadata"]
            assert ("能量", "280 kJ") in detail["nutrition_rows"]
            assert ("蛋白质", "3.2 g") in detail["nutrition_rows"]
        finally:
            db.close()


def test_inventory_detail_for_fresh_food_uses_builtin_nutrition():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("apple", 2)

            detail = demo.build_inventory_detail(db, "apple")
            joined = "\n".join(detail["lines"])

            assert detail["status"] == "ok"
            assert "膳食纤维" in joined
            assert "维生素C" in joined
            assert "无需标签录入" in joined
            assert ("来源", "本地默认营养库") in detail["metadata"]
            assert ("膳食纤维", "约2.4 g / 100g") in detail["nutrition_rows"]
        finally:
            db.close()


def test_inventory_page_registers_six_food_detail_buttons():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            demo.draw_inventory_page(
                db=db,
                detect_counts=demo.empty_counts(),
                stable_counts=demo.empty_counts(),
                db_counts=demo.empty_counts(),
                fps=0,
                infer_ms=0,
                stable_votes=0,
                frame_quality=None,
            )

            actions = {button["action"] for button in demo.UI_STATE["buttons"]}
            expected = {f"inventory_select_{name}" for name in demo.CLASS_ORDER}
            assert expected <= actions
        finally:
            db.close()


def test_label_record_is_cleared_when_packaged_food_count_reaches_zero():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)
            record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            db.save_label_record(record)
            assert db.get_latest_label_record("milk_box") is not None

            db.set_count("milk_box", 0)

            assert db.get_latest_label_record("milk_box") is None
            milk = next(item for item in db.get_inventory_items() if item["class_name"] == "milk_box")
            assert milk["latest_label"] is None
        finally:
            db.close()


def test_packaged_label_record_is_cleared_when_food_is_readded_without_new_label():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)
            record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            db.save_label_record(record)
            assert db.get_latest_label_record("milk_box") is not None

            db.set_count("milk_box", 0)
            db.set_count("milk_box", 1)

            assert db.get_latest_label_record("milk_box") is None
            detail = demo.build_inventory_detail(db, "milk_box")
            assert detail["status"] == "missing_label"
        finally:
            db.close()


def test_packaged_label_record_saved_before_stock_is_not_used_after_new_stock_arrives():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            db.save_label_record(record)
            assert db.get_counts()["milk_box"] == 0
            assert db.get_latest_label_record("milk_box") is not None

            db.set_count("milk_box", 1)

            assert db.get_latest_label_record("milk_box") is None
            detail = demo.build_inventory_detail(db, "milk_box")
            assert detail["status"] == "missing_label"
        finally:
            db.close()


def test_reset_zero_clears_active_label_records():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)
            db.set_count("coke_can", 1)
            db.save_label_record(demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"]))
            db.save_label_record(demo.parse_label_ocr_text("coke_can", demo.LABEL_SAMPLE_OCR_TEXT["coke_can"]))

            db.reset_zero()

            assert db.get_latest_label_record("milk_box") is None
            assert db.get_latest_label_record("coke_can") is None
        finally:
            db.close()


def test_reset_then_same_stable_snapshot_readds_inventory_and_speaks():
    demo = load_demo_module()

    class FakeSpeaker:
        def __init__(self):
            self.messages = []

        def say(self, text, prompt_key=None):
            self.messages.append(text)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        speaker = FakeSpeaker()
        try:
            db.set_count("milk_box", 1)
            db.reset_zero()

            stable_counts = demo.empty_counts()
            stable_counts["milk_box"] = 1
            demo.apply_inventory_change(db, speaker, stable_counts, {"milk_box": 0.0})

            assert db.get_counts()["milk_box"] == 1
            assert speaker.messages == ["盒装牛奶增加1盒，当前1盒"]
        finally:
            db.close()


def test_save_label_capture_record_saves_record_and_speaks_success():
    demo = load_demo_module()

    class FakeSpeaker:
        def __init__(self):
            self.messages = []

        def say(self, text, prompt_key=None):
            self.messages.append((text, prompt_key))

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        speaker = FakeSpeaker()
        try:
            image_path = str(Path(tmp_dir) / "label.jpg")
            ok, record = demo.save_label_capture_record(db, speaker, "coke_can", image_path)

            assert ok is True
            assert record["class_name"] == "coke_can"
            assert record["image_path"] == image_path
            latest = db.get_latest_label_record("coke_can")
            assert latest["product_name"] == "罐装可乐"
            assert speaker.messages == [("标签录入成功，已记录保质期和营养信息", "label_saved")]
        finally:
            db.close()


def test_save_label_capture_record_requires_capture_path():
    demo = load_demo_module()

    class FakeSpeaker:
        def __init__(self):
            self.messages = []

        def say(self, text, prompt_key=None):
            self.messages.append((text, prompt_key))

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        speaker = FakeSpeaker()
        try:
            ok, record = demo.save_label_capture_record(db, speaker, "milk_box", "")

            assert ok is False
            assert record is None
            assert db.get_latest_label_record("milk_box") is None
            assert speaker.messages == [("标签识别失败，请重新对准包装标签", "label_failed")]
        finally:
            db.close()


def test_assistant_page_title_omits_mode_and_ready_suffix():
    demo = load_demo_module()
    np = __import__("numpy")

    class FakeLLM:
        def available(self):
            return False

    class FakeAssistant:
        llm_client = FakeLLM()
        busy = False
        latest_question = ""
        latest_answer = "助手已就绪。"

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        captured_text = []
        original_put_text = demo.put_text
        try:
            demo.put_text = lambda img, text, *args, **kwargs: captured_text.append(str(text))
            demo.draw_assistant_page(
                db=db,
                assistant=FakeAssistant(),
                fps=0,
                infer_ms=0,
                stable_votes=0,
                frame_quality=None,
            )
        finally:
            demo.put_text = original_put_text
            db.close()

    assert "助手回答" in captured_text
    assert all("本地规则模式" not in text for text in captured_text)
    assert all("就绪)" not in text for text in captured_text)


def wait_until(predicate, timeout=2.0):
    demo = load_demo_module()
    deadline = demo.time.time() + timeout
    while demo.time.time() < deadline:
        if predicate():
            return True
        demo.time.sleep(0.02)
    return False


def test_apply_inventory_change_requests_cloud_analysis_once_for_batch_update():
    demo = load_demo_module()

    class FakeSpeaker:
        def __init__(self):
            self.messages = []

        def say(self, text, prompt_key=None):
            self.messages.append((text, prompt_key))

    class FakeCloudAnalysis:
        def __init__(self):
            self.reasons = []

        def request_analysis(self, reason="manual"):
            self.reasons.append(reason)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            speaker = FakeSpeaker()
            cloud_analysis = FakeCloudAnalysis()
            stable_counts = demo.empty_counts()
            stable_counts["milk_box"] = 1
            stable_counts["coke_can"] = 1

            demo.apply_inventory_change(
                db=db,
                speaker=speaker,
                stable_counts=stable_counts,
                last_event_time={name: 0.0 for name in demo.CLASS_ORDER},
                cloud_analysis=cloud_analysis,
            )

            assert cloud_analysis.reasons == ["inventory_change"]
            assert len(speaker.messages) == 2
        finally:
            db.close()


def test_cloud_analysis_manager_sends_inventory_labels_and_events_to_llm():
    demo = load_demo_module()

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def available(self):
            return True

        def ask(self, question, inventory_payload):
            self.calls.append((question, inventory_payload))
            return "云端建议：优先饮用牛奶，饮料适量。"

    class FakeSpeaker:
        def __init__(self):
            self.messages = []

        def say(self, text, prompt_key=None):
            self.messages.append((text, prompt_key))

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)
            record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            db.save_label_record(record)
            db.log_event("milk_box", 1, 0, 1)
            llm = FakeLLM()
            manager = demo.CloudAnalysisManager(
                db=db,
                speaker=FakeSpeaker(),
                llm_client=llm,
                debounce_seconds=0.0,
            )

            manager.request_analysis(reason="manual")

            assert wait_until(lambda: manager.get_state()["status"] == "done")
            state = manager.get_state()
            assert state["source"] == "cloud"
            assert "云端建议" in state["answer"]
            assert len(llm.calls) == 1
            question, payload = llm.calls[0]
            assert "饮食建议" in question
            milk = next(item for item in payload["inventory"] if item["class_name"] == "milk_box")
            assert milk["latest_label"]["product_name"] == "纯牛奶"
            assert payload["recent_events"][0]["class_name"] == "milk_box"
        finally:
            db.close()


def test_cloud_analysis_payload_excludes_zero_count_label_records():
    demo = load_demo_module()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("milk_box", 1)
            milk_record = demo.parse_label_ocr_text("milk_box", demo.LABEL_SAMPLE_OCR_TEXT["milk_box"])
            coke_record = demo.parse_label_ocr_text("coke_can", demo.LABEL_SAMPLE_OCR_TEXT["coke_can"])
            db.save_label_record(milk_record)
            db.save_label_record(coke_record)
            db.set_count("coke_can", 0)

            payload = demo.build_cloud_analysis_payload(db)

            classes = [item["class_name"] for item in payload["inventory"]]
            assert classes == ["milk_box"]
            assert payload["inventory"][0]["latest_label"]["product_name"] == "纯牛奶"
        finally:
            db.close()


def test_cloud_analysis_manager_falls_back_when_llm_has_no_answer():
    demo = load_demo_module()

    class EmptyLLM:
        def available(self):
            return True

        def ask(self, question, inventory_payload):
            return ""

    class FakeSpeaker:
        def say(self, text, prompt_key=None):
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            db.set_count("banana", 1)
            manager = demo.CloudAnalysisManager(
                db=db,
                speaker=FakeSpeaker(),
                llm_client=EmptyLLM(),
                debounce_seconds=0.0,
            )

            manager.request_analysis(reason="manual")

            assert wait_until(lambda: manager.get_state()["status"] == "fallback")
            state = manager.get_state()
            assert state["source"] == "local_fallback"
            assert "香蕉" in state["answer"]
        finally:
            db.close()


def test_cloud_analysis_manager_ignores_stale_analysis_result():
    demo = load_demo_module()

    class SlowFirstLLM:
        def __init__(self):
            self.calls = 0
            self.first_started = demo.threading.Event()
            self.release_first = demo.threading.Event()

        def available(self):
            return True

        def ask(self, question, inventory_payload):
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                self.release_first.wait(1.0)
                return "旧库存建议"
            return "新库存建议"

    class FakeSpeaker:
        def say(self, text, prompt_key=None):
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            llm = SlowFirstLLM()
            manager = demo.CloudAnalysisManager(
                db=db,
                speaker=FakeSpeaker(),
                llm_client=llm,
                debounce_seconds=0.0,
            )

            manager.request_analysis(reason="inventory_change")
            assert llm.first_started.wait(1.0)
            manager.request_analysis(reason="inventory_change")
            llm.release_first.set()

            assert wait_until(lambda: manager.get_state()["answer"] == "新库存建议")
            assert manager.get_state()["answer"] != "旧库存建议"
        finally:
            db.close()


def test_cloud_voice_question_auto_speaks_answer():
    demo = load_demo_module()

    class FakeLLM:
        def available(self):
            return True

        def ask(self, question, inventory_payload):
            return "可以先吃香蕉，再喝牛奶。"

    class FakeSpeaker:
        def __init__(self):
            self.messages = []

        def say(self, text, prompt_key=None):
            self.messages.append((text, prompt_key))

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            speaker = FakeSpeaker()
            manager = demo.CloudAnalysisManager(
                db=db,
                speaker=speaker,
                llm_client=FakeLLM(),
                debounce_seconds=0.0,
            )

            manager.request_voice_question("请结合当前库存给出饮食建议")

            assert wait_until(lambda: manager.get_state()["voice_status"] == "done")
            state = manager.get_state()
            assert state["voice_question"] == "请结合当前库存给出饮食建议"
            assert state["voice_answer"] == "可以先吃香蕉，再喝牛奶。"
            assert speaker.messages == [("可以先吃香蕉，再喝牛奶。", None)]
        finally:
            db.close()


def test_assistant_page_registers_cloud_touch_buttons():
    demo = load_demo_module()
    np = __import__("numpy")

    class FakeLLM:
        def available(self):
            return True

    class FakeAssistant:
        llm_client = FakeLLM()
        busy = False
        latest_question = ""
        latest_answer = "助手已就绪。"

    class FakeCloudAnalysis:
        def get_state(self):
            return {
                "status": "idle",
                "status_text": "等待云端分析",
                "source": "",
                "answer": "暂无云端建议。",
                "voice_status": "idle",
                "voice_question": "",
                "voice_answer": "",
            }

    with tempfile.TemporaryDirectory() as tmp_dir:
        db = demo.InventoryDB(str(Path(tmp_dir) / "inventory.db"))
        try:
            canvas = demo.draw_assistant_page(
                db=db,
                assistant=FakeAssistant(),
                cloud_analysis=FakeCloudAnalysis(),
                fps=0,
                infer_ms=0,
                stable_votes=0,
                frame_quality=None,
            )
            assert canvas.shape == (demo.UI_H, demo.UI_W, 3)
            actions = {button["action"] for button in demo.UI_STATE["buttons"]}
            assert {"cloud_analyze", "speak_cloud_advice", "voice_question"} <= actions
        finally:
            db.close()


if __name__ == "__main__":
    test_chinese_speech_and_inventory_answer()
    test_measure_text_supports_old_pillow_font_api()
    test_chinese_text_sprite_is_cached()
    test_put_text_uses_baseline_for_chinese_sprite()
    test_header_registers_quit_button()
    test_header_registers_label_page_button()
    test_ui_recorder_writes_frames()
    test_normalize_wav_file_converts_mono_44k_to_stereo_48k()
    test_speaker_prefers_wav_prompt_when_available()
    test_speaker_falls_back_to_tts_without_wav_prompt()
    test_speaker_builds_tts_fallback_after_wav_prompt()
    test_speaker_uses_explicit_prompt_key_for_wav_prompt()
    test_speaker_retries_wav_once_before_tts_fallback()
    test_speaker_falls_back_to_tts_after_wav_retries_fail()
    test_speaker_times_out_stuck_commands_and_continues_queue()
    test_speaker_builds_hdmi_audio_init_command_for_plughw_device()
    test_speaker_uses_event_wav_for_inventory_addition()
    test_speaker_uses_event_wav_for_inventory_removal()
    test_speaker_uses_blueberry_half_wav_for_half_box_change()
    test_speaker_uses_inventory_empty_wav_for_empty_answer()
    test_speaker_uses_reminder_saved_wav_for_dynamic_reminder_answer()
    test_estimates_blueberry_half_box_from_blue_area()
    test_estimates_blueberry_full_box_from_binary_dark_area()
    test_rejects_blueberry_shadow_without_blue_purple_content()
    test_postprocess_drops_ambiguous_milk_coke_classification()
    test_dark_frame_quality_freezes_inventory_update()
    test_open_camera_falls_back_to_available_device()
    test_runtime_config_overrides_tunable_parameters()
    test_runtime_config_resolves_relative_wav_dir_from_config_file()
    test_project_config_uses_fixed_board_camera_and_valid_deepseek_model()
    test_llm_client_logs_http_error_response_body()
    test_preprocess_suppresses_plastic_wrap_glare()
    test_parse_milk_sample_ocr_extracts_label_fields()
    test_parse_coke_sample_ocr_extracts_label_fields()
    test_label_records_are_saved_and_latest_record_is_returned()
    test_inventory_answer_uses_latest_label_record_for_packaged_food()
    test_inventory_detail_for_packaged_food_requires_label_record()
    test_inventory_detail_for_packaged_food_uses_label_nutrition()
    test_inventory_detail_for_fresh_food_uses_builtin_nutrition()
    test_inventory_page_registers_six_food_detail_buttons()
    test_label_record_is_cleared_when_packaged_food_count_reaches_zero()
    test_packaged_label_record_is_cleared_when_food_is_readded_without_new_label()
    test_packaged_label_record_saved_before_stock_is_not_used_after_new_stock_arrives()
    test_reset_zero_clears_active_label_records()
    test_reset_then_same_stable_snapshot_readds_inventory_and_speaks()
    test_save_label_capture_record_saves_record_and_speaks_success()
    test_save_label_capture_record_requires_capture_path()
    test_assistant_page_title_omits_mode_and_ready_suffix()
    test_apply_inventory_change_requests_cloud_analysis_once_for_batch_update()
    test_cloud_analysis_manager_sends_inventory_labels_and_events_to_llm()
    test_cloud_analysis_payload_excludes_zero_count_label_records()
    test_cloud_analysis_manager_falls_back_when_llm_has_no_answer()
    test_cloud_analysis_manager_ignores_stale_analysis_result()
    test_cloud_voice_question_auto_speaks_answer()
    test_assistant_page_registers_cloud_touch_buttons()
    print("assistant tests passed")
