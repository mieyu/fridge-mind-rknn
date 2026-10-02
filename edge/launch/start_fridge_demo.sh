#!/usr/bin/env bash
set -u

if [ "${1:-}" = "--in-terminal" ]; then
  export FRIDGE_LAUNCHER_IN_TERMINAL=1
  shift
fi

SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")"
if [ "${FRIDGE_LAUNCHER_IN_TERMINAL:-0}" != "1" ] && [ ! -t 1 ]; then
  export FRIDGE_LAUNCHER_IN_TERMINAL=1
  TERM_CMD="$(printf '%q --in-terminal' "$SCRIPT_PATH")"
  for term in x-terminal-emulator gnome-terminal xfce4-terminal lxterminal konsole mate-terminal xterm; do
    if command -v "$term" >/dev/null 2>&1; then
      case "$term" in
        gnome-terminal|mate-terminal)
          exec "$term" -- bash -lc "$TERM_CMD"
          ;;
        xfce4-terminal)
          exec "$term" --hold -e "bash -lc $TERM_CMD"
          ;;
        lxterminal)
          exec "$term" -e bash -lc "$TERM_CMD"
          ;;
        konsole)
          exec "$term" --hold -e bash -lc "$TERM_CMD"
          ;;
        x-terminal-emulator|xterm)
          exec "$term" -e bash -lc "$TERM_CMD"
          ;;
      esac
    fi
  done
fi

APP_DIR="${FRIDGE_APP_DIR:-/home/ztl/fridge_project/scripts}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MAIN_SCRIPT="camera_inventory_ui_demo.py"
CONFIG_FILE="fridge_demo_config.json"
WAV_DIR="voice_prompts"
CAMERA_DEVICE="${FRIDGE_CAMERA_DEVICE:-/dev/video18}"
AUDIO_DEVICE="${FRIDGE_AUDIO_DEVICE:-plughw:1,0}"
INTERVAL="${FRIDGE_INFER_INTERVAL:-8}"
PYTHON_BIN="${FRIDGE_PYTHON_BIN:-python3}"
MODEL_FP="/home/ztl/fridge_project/model/best_fp.rknn"
MODEL_INT8="/home/ztl/fridge_project/model/best.rknn"
CLASSES_FILE="/home/ztl/fridge_project/model/classes.txt"
NORMALIZED_READY_WAV="/tmp/fridge_start_system_ready_48k.wav"

if [ ! -d "$APP_DIR" ] && [ -f "$SCRIPT_DIR/$MAIN_SCRIPT" ]; then
  APP_DIR="$SCRIPT_DIR"
fi

pause() {
  printf '\n'
  read -r -p "Press Enter to close this window..." _
}

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*"
}

fail() {
  printf '[FAIL] %s\n' "$*"
  pause
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

require_file() {
  [ -f "$1" ] || fail "Required file not found: $1"
}

require_dir() {
  [ -d "$1" ] || fail "Required directory not found: $1"
}

cd "$APP_DIR" 2>/dev/null || fail "Cannot enter app directory: $APP_DIR"

clear
info "RK3566 smart fridge demo launcher"
info "App directory: $APP_DIR"
info "Camera: $CAMERA_DEVICE"
info "Audio: $AUDIO_DEVICE"
info "Infer interval: $INTERVAL"
printf '\n'

info "Stopping leftover speaker-test process if present..."
pkill speaker-test 2>/dev/null || true

info "Checking required commands..."
require_command "$PYTHON_BIN"
require_command aplay
require_command amixer
if ! command -v espeak-ng >/dev/null 2>&1; then
  warn "espeak-ng not found. Fixed WAV prompts can still play, but dynamic TTS will fail."
fi

info "Checking project files..."
require_file "$MAIN_SCRIPT"
require_file "$CONFIG_FILE"
require_dir "$WAV_DIR"
require_file "$WAV_DIR/system_ready.wav"
require_file "$CLASSES_FILE"
if [ ! -f "$MODEL_FP" ] && [ ! -f "$MODEL_INT8" ]; then
  fail "No RKNN model found: $MODEL_FP or $MODEL_INT8"
fi

wav_count="$(find "$WAV_DIR" -maxdepth 1 -type f -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
info "WAV prompt count: $wav_count"
if [ "${wav_count:-0}" -lt 30 ]; then
  warn "Expected 30 WAV prompt files. Some prompts may fall back to TTS."
fi

info "Checking camera device..."
if [ ! -e "$CAMERA_DEVICE" ]; then
  fail "Camera device does not exist: $CAMERA_DEVICE"
fi
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --device="$CAMERA_DEVICE" --all 2>/dev/null | sed -n '1,18p' || true
else
  warn "v4l2-ctl not found. Skipping detailed camera capability check."
fi

info "Initializing HDMI audio..."
amixer -c 1 sset 'ELD Bypass' on >/dev/null 2>&1 || warn "Failed to set HDMI ELD Bypass; continuing."

info "Testing normalized startup WAV..."
"$PYTHON_BIN" "$MAIN_SCRIPT" --normalize-audio "$WAV_DIR/system_ready.wav" "$NORMALIZED_READY_WAV" \
  || fail "Audio normalization failed."
aplay -D "$AUDIO_DEVICE" "$NORMALIZED_READY_WAV" \
  || fail "Audio playback failed on $AUDIO_DEVICE."

info "Current key config lines:"
grep -n '"device"\|"audio_device"\|"model"' "$CONFIG_FILE" || true
printf '\n'

if pgrep -f "$MAIN_SCRIPT" >/dev/null 2>&1; then
  warn "Another camera_inventory_ui_demo.py process appears to be running."
  warn "If the camera cannot open, close the old process and run this launcher again."
fi

info "Starting main demo. Press q in the UI to exit."
printf '\n'
"$PYTHON_BIN" -u "$MAIN_SCRIPT" --camera "$CAMERA_DEVICE" --fullscreen --interval "$INTERVAL"
status=$?
printf '\n'
info "Demo process exited with status: $status"
pause
exit "$status"
