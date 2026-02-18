


import os
import cv2
import threading
import time
import logging
import json
import re
from datetime import datetime, timezone
from collections import deque
try:
    from google import genai
except Exception:
    genai = None
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
from ultralytics import YOLO


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov9c.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "app3", "model", "yolov9c.pt")
model = YOLO(MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass

LOG_MAX_LINES = 1500
LOG_PATH = os.path.join(PROJECT_ROOT, "app.log")
STRUCT_LOG_PATH = os.path.join(PROJECT_ROOT, "run_structured.jsonl")
logger = logging.getLogger("thermal_cam")
logger.setLevel(logging.INFO)
logger.handlers = []
logger.propagate = False

file_handler = None
struct_log_file = None
log_buffer = deque(maxlen=LOG_MAX_LINES)
log_seq = 0
log_lock = threading.Lock()
tts_engine = None
tts_queue = deque()
tts_event = threading.Event()
tts_lock = threading.Lock()
tts_worker = None


def reset_logging():
    global file_handler, struct_log_file, log_buffer, log_seq
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    if struct_log_file:
        try:
            struct_log_file.close()
        except Exception:
            pass
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    struct_log_file = open(STRUCT_LOG_PATH, "w", encoding="utf-8")
    with log_lock:
        log_buffer.clear()
        log_seq = 0


reset_logging()


def _enforce_log_file_limit():
    # Keep persistent log capped at LOG_MAX_LINES entries.
    if LOG_MAX_LINES <= 0 or not os.path.exists(LOG_PATH):
        return
    try:
        if file_handler:
            file_handler.flush()
        with open(LOG_PATH, "r+", encoding="utf-8") as fh:
            lines = fh.readlines()
            if len(lines) <= LOG_MAX_LINES:
                return
            fh.seek(0)
            fh.writelines(lines[-LOG_MAX_LINES:])
            fh.truncate()
    except Exception:
        pass


def log(message):
    global log_seq
    logger.info(message)
    with log_lock:
        log_seq += 1
        log_buffer.append((log_seq, f"{datetime.now().strftime('%H:%M:%S')} | {message}"))
        if log_seq > LOG_MAX_LINES:
            _enforce_log_file_limit()


def log_structured(payload):
    if not struct_log_file:
        return
    struct_log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    struct_log_file.flush()


def _format_person_id(frame_id, detection_index, track_id):
    if track_id not in (None, "", "None"):
        try:
            return f"{int(track_id):02d}"
        except Exception:
            return str(track_id)
    fallback_idx = detection_index if detection_index is not None else "x"
    base_frame = "unknown" if frame_id is None else frame_id
    return f"det-{base_frame}-{fallback_idx}"


def _normalize_struct_event(payload):
    if not isinstance(payload, dict):
        return None
    person_id = payload.get("person_id")
    det_idx = payload.get("detection_index")
    frame_id = payload.get("frame_id", "unknown")
    if person_id in (None, "", "None"):
        payload["person_id"] = _format_person_id(frame_id, det_idx, None)
    else:
        payload["person_id"] = str(person_id)
    return payload


def init_tts():
    global tts_engine, tts_worker
    if tts_engine is not None or pyttsx3 is None:
        return tts_engine
    try:
        engine = pyttsx3.init(driverName="sapi5")
        engine.setProperty("rate", 185)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices") or []
        for voice in voices:
            name = getattr(voice, "name", "")
            if isinstance(name, str) and "zira" in name.lower():
                engine.setProperty("voice", voice.id)
                break
        tts_engine = engine
        tts_worker = threading.Thread(target=_tts_loop, daemon=True)
        tts_worker.start()
    except Exception as exc:
        tts_engine = None
        log(f"TTS disabled: {exc}")
    return tts_engine


def _tts_loop():
    while True:
        tts_event.wait()
        while True:
            with tts_lock:
                if not tts_queue:
                    tts_event.clear()
                    break
                text = tts_queue.popleft()
            try:
                if tts_engine is None:
                    break
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as exc:
                log(f"TTS error: {exc}")
                break


def _sanitize_for_tts(text):
    if not text:
        return ""
    # Reduce to alphabetic text so the voice module skips numbers/special chars.
    cleaned = re.sub(r"[^A-Za-z\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def speak_agent_text(message):
    if not message:
        return
    spoken = _sanitize_for_tts(message)
    if not spoken:
        return
    if init_tts() is None:
        return
    with tts_lock:
        tts_queue.append(spoken)
    tts_event.set()


def setup_agent():
    if not AGENT_ENABLED:
        return None
    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        log("Agent disabled: missing GOOGLE_API_KEY/GEMINI_API_KEY.")
        return None
    if genai is None:
        log("Agent disabled: google-genai package not available.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        log(f"Agent disabled: {exc}")
        return None


def agent_analyze(client, log_text):
    global agent_health
    if not client or not log_text:
        return None
    prompt = (
        "You monitor a thermal security feed. Using the recent human-activity log, "
        "write a short descriptive update (max four sentences or bullet points). "
        "Summaries should highlight person IDs, motion state (standing/moving), with frame id and time stamp currnt any "
        "risks (loitering, spikes), and reference timestamps or frame IDs for "
        "context. Prefer Markdown lists when multiple people are involved. Avoid "
        "copying the log verbatim—synthesize it into natural language. If nothing "
        "important is happening, reply OK.\n\n"
        f"Event digest:\n{log_text}"
    )
    try:
        response = client.models.generate_content(model=AGENT_MODEL, contents=prompt)
        agent_text = response.text.strip() if response and response.text else None
        if agent_text:
            normalized = agent_text.strip()
            if normalized.upper() == "OK":
                agent_health["last_summary"] = "No significant human activity detected."
            else:
                agent_health["last_summary"] = agent_text
            agent_health["last_error"] = None
            agent_health["last_success_ts"] = time.time()
            speak_agent_text(agent_text)
        return agent_text
    except Exception as exc:
        error_str = str(exc).lower()
        # Check if it's a rate limit or quota error
        is_rate_limit = any(keyword in error_str for keyword in [
            'rate limit', 'quota', '429', 'resourceexhausted', 
            'too many requests', 'rate_limit_exceeded'
        ])
        
        if is_rate_limit:
            agent_health["last_error"] = f"Rate limit exceeded - using raw logs"
            log(f"AGENT rate limit exceeded, showing raw activity digest")
            # Return the raw log text as fallback
            raw_summary = f"[RAW ACTIVITY LOG]\n{log_text}"
            agent_health["last_summary"] = raw_summary
            speak_agent_text("Rate limit exceeded. Showing raw activity log.")
            return raw_summary
        else:
            agent_health["last_error"] = str(exc)
            log(f"AGENT error: {exc}")
            return None


def agent_watch_loop():
    global agent_health
    if not agent_client:
        agent_health["online"] = False
        return
    agent_health["online"] = True
    try:
        last_processed_marker = None
        first_activity_detected_at = None
        pending_summary = False
        last_report_ts = 0.0
        last_mtime = 0.0
        while not stop_event.is_set():
            try:
                current_mtime = os.path.getmtime(STRUCT_LOG_PATH)
            except OSError:
                current_mtime = 0.0
            if current_mtime <= last_mtime and not pending_summary:
                time.sleep(0.5)
                continue
            if current_mtime:
                last_mtime = current_mtime
            events = read_structured_full(person_only=True)
            if not events:
                pending_summary = False
                first_activity_detected_at = None
                time.sleep(0.5)
                continue
            latest_event = events[-1]
            marker = f"{latest_event.get('timestamp')}|{latest_event.get('frame_id')}"
            if marker != last_processed_marker and not pending_summary:
                pending_summary = True
                first_activity_detected_at = time.time()
            if not pending_summary:
                time.sleep(0.5)
                continue
            now_ts = time.time()
            if first_activity_detected_at and (now_ts - first_activity_detected_at) < AGENT_EVERY_SEC:
                time.sleep(0.5)
                continue
            if last_report_ts and (now_ts - last_report_ts) < AGENT_EVERY_SEC:
                time.sleep(0.5)
                continue
            digest = build_activity_digest(events)
            if not digest:
                pending_summary = False
                time.sleep(0.5)
                continue
            agent_text = agent_analyze(agent_client, digest)
            if agent_text and agent_text.upper() != "OK":
                log(f"AGENT | {agent_text}")
                print(f"AGENT | {agent_text}")
            last_processed_marker = marker
            pending_summary = False
            last_report_ts = now_ts
            time.sleep(0.5)
    except Exception as exc:
        agent_health["online"] = False
        agent_health["last_error"] = str(exc)
        log(f"AGENT watch error: {exc}")
    finally:
        agent_health["online"] = False

RESIZE_WIDTH = 384

INFER_EVERY_N = 3

MAX_INFER_FPS = 5
EXPLAIN_WIDTH = 320
EXPLAIN_EVERY_N = 3
DENSITY_WINDOW_SEC = 5.0
LOITER_WINDOW_SEC = 5.0

TRACK_MAX_MISSES = 15
TRACK_MATCH_MAX_DIST = 80
STAND_SPEED_PX = 3.0
THERMAL_HISTORY_SEC = 5.0
THERMAL_SPIKE_K = 2.5
THERMAL_MIN_DELTA = 20.0
STANDING_ALERT_SEC = 8.0
EDGE_MARGIN_PX = 40
ALERT_COOLDOWN_SEC = 5.0
AGENT_ENABLED = True
AGENT_MODEL = "gemini-2.5-flash"
AGENT_EVERY_SEC = 30.0
AGENT_MAX_EVENTS = 25

agent_client = None
agent_health = {
    "enabled": AGENT_ENABLED,
    "online": False,
    "last_error": None,
    "last_summary": None,
    "last_success_ts": 0.0,
}

latest_lock = threading.Lock()
latest_infer = {"id": -1, "frame": None, "orig": None, "scale": (1.0, 1.0)}
latest_annotated = {"id": -1, "color": None, "explain": None}
explain_cache = {"frame_id": -1, "frame": None}
stop_event = threading.Event()
worker = None
agent_thread = None
capture_thread = None
display_delay_ms = 33
processing_lock = threading.Lock()
processing_state = {"running": False, "video_path": None, "error": None}

WINDOW_NAME_COLOR = "YOLOv9 Thermal"
WINDOW_NAME_EXPLAIN = "YOLOv9 Explain"
SHOW_EXPLAIN = True
TEXT_SCALE = 0.4
TEXT_THICKNESS = 1
BBOX_COLOR = (255, 255, 255)
BBOX_THICKNESS = 1
DASH_LEN = 6
DASH_GAP = 4
COLORMAP = cv2.COLORMAP_INFERNO


def apply_thermal_colormap(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, COLORMAP)


def draw_detections(base_frame, results, scale_x, scale_y, per_det_info=None):
    annotated = base_frame.copy() if base_frame is not None else None
    if annotated is None:
        return base_frame
    boxes = results.boxes
    if boxes is None or boxes.xyxy is None:
        return annotated

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
    clses = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        cls_id = int(clses[i]) if len(clses) > i else -1
        conf = float(confs[i]) if len(confs) > i else 0.0
        if isinstance(model.names, dict):
            label = model.names.get(cls_id, "obj")
        else:
            label = model.names[cls_id] if 0 <= cls_id < len(model.names) else "obj"
        bw = max(0, x2 - x1)
        suffix = ""
        if per_det_info and i in per_det_info and label.lower() == "person":
            track_id, state = per_det_info[i]
            suffix = f" id={track_id} {state}"
        text = f"{label}{suffix} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

    
        cy = (y1 + y2) // 2
        x = x1
        while x < x2:
            x_end = min(x + DASH_LEN, x2)
            cv2.line(annotated, (x, cy), (x_end, cy), BBOX_COLOR, 1)
            x += DASH_LEN + DASH_GAP

        # Confidence bar (small, under label area)
        bar_w = max(40, min(100, bw))
        bar_h = 4
        bar_x1 = x1
        bar_y1 = min(annotated.shape[0] - bar_h - 1, y1 + 2)
        bar_x2 = bar_x1 + bar_w
        bar_y2 = bar_y1 + bar_h
        fill_w = int(bar_w * max(0.0, min(1.0, conf)))
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 1)
        if fill_w > 0:
            cv2.rectangle(
                annotated,
                (bar_x1, bar_y1),
                (bar_x1 + fill_w, bar_y2),
                BBOX_COLOR,
                -1,
            )
        label_lower = label.lower()
        if label_lower == "car":
            text_color = (255, 255, 255)
        elif label_lower == "person":
            text_color = (0, 255, 255)
        else:
            text_color = (0, 255, 0)

        cv2.putText(
            annotated,
            text,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            text_color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


class PersonTrack:
    def __init__(self, track_id, cx, cy, ts):
        self.id = track_id
        self.last_cx = cx
        self.last_cy = cy
        self.last_ts = ts
        self.misses = 0
        self.history = deque()
        self.history.append((ts, cx, cy))
        self.state = "standing"
        self.standing_since = ts
        self.thermal_history = deque()
        self.last_alert_ts = {}

    def update(self, cx, cy, ts):
        self.last_cx = cx
        self.last_cy = cy
        self.last_ts = ts
        self.misses = 0
        self.history.append((ts, cx, cy))
        while self.history and (ts - self.history[0][0]) > LOITER_WINDOW_SEC:
            self.history.popleft()

    def mark_missed(self):
        self.misses += 1

    def compute_speed(self):
        if len(self.history) < 2:
            return 0.0
        (t0, x0, y0) = self.history[0]
        (t1, x1, y1) = self.history[-1]
        dt = max(1e-6, t1 - t0)
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return dist / dt

    def update_state(self):
        speed = self.compute_speed()
        if speed < STAND_SPEED_PX:
            if self.state != "standing":
                self.standing_since = self.last_ts
            self.state = "standing"
        else:
            self.state = "moving"
            self.standing_since = None

    def add_thermal(self, ts, value):
        self.thermal_history.append((ts, value))
        while self.thermal_history and (ts - self.thermal_history[0][0]) > THERMAL_HISTORY_SEC:
            self.thermal_history.popleft()

    def thermal_stats(self):
        if not self.thermal_history:
            return 0.0, 0.0
        values = [v for _, v in self.thermal_history]
        mean = sum(values) / float(len(values))
        var = sum((v - mean) ** 2 for v in values) / float(len(values))
        return mean, var ** 0.5

    def can_alert(self, key, ts):
        last_ts = self.last_alert_ts.get(key, 0.0)
        if ts - last_ts >= ALERT_COOLDOWN_SEC:
            self.last_alert_ts[key] = ts
            return True
        return False


def is_near_edge(cx, cy, frame_w, frame_h):
    return (
        cx < EDGE_MARGIN_PX
        or cy < EDGE_MARGIN_PX
        or cx > (frame_w - EDGE_MARGIN_PX)
        or cy > (frame_h - EDGE_MARGIN_PX)
    )


def mean_box_intensity(gray_frame, box):
    if gray_frame is None:
        return 0.0
    x1, y1, x2, y2 = box
    h, w = gray_frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(roi.mean())


def box_intensity_stats(gray_frame, box):
    if gray_frame is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = box
    h, w = gray_frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0
    return float(roi.mean()), float(roi.max())


def draw_explainability(base_frame, gray_frame, results, scale_x, scale_y):
    if base_frame is None or gray_frame is None:
        return base_frame
    explain = base_frame.copy()
    boxes = results.boxes
    if boxes is None or boxes.xyxy is None:
        return explain

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
    clses = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
    edges = cv2.Canny(gray_frame, 50, 120)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for i in range(len(xyxy)):
        cls_id = int(clses[i]) if len(clses) > i else -1
        if isinstance(model.names, dict):
            label = model.names.get(cls_id, "obj")
        else:
            label = model.names[cls_id] if 0 <= cls_id < len(model.names) else "obj"
        is_person = label.lower() == "person"

        x1, y1, x2, y2 = xyxy[i]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        h, w = gray_frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = gray_frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        heat = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
        heat_color = cv2.applyColorMap(heat.astype("uint8"), cv2.COLORMAP_JET)
        base_roi = explain[y1:y2, x1:x2]
        explain[y1:y2, x1:x2] = cv2.addWeighted(base_roi, 0.55, heat_color, 0.45, 0)

        edge_roi = edges_color[y1:y2, x1:x2]
        explain[y1:y2, x1:x2] = cv2.addWeighted(
            explain[y1:y2, x1:x2], 0.85, edge_roi, 0.15, 0
        )

        if is_person:
            band_h = max(1, (y2 - y1) // 3)
            torso_roi = roi[band_h : 2 * band_h, :]
            head_roi = roi[:band_h, :]
            legs_roi = roi[2 * band_h :, :]
            torso_mean = float(torso_roi.mean()) if torso_roi.size else 0.0
            head_mean = float(head_roi.mean()) if head_roi.size else 0.0
            legs_mean = float(legs_roi.mean()) if legs_roi.size else 0.0
            torso_hint = "torso heat" if torso_mean >= max(head_mean, legs_mean) else ""

            left_roi = roi[:, : (roi.shape[1] // 2)]
            right_roi = roi[:, (roi.shape[1] // 2) :]
            left_mean = float(left_roi.mean()) if left_roi.size else 0.0
            right_mean = float(right_roi.mean()) if right_roi.size else 0.0
            if max(left_mean, right_mean) > 0:
                symmetry = 1.0 - abs(left_mean - right_mean) / max(left_mean, right_mean)
            else:
                symmetry = 0.0
            symmetry_hint = "limb symmetry" if symmetry > 0.8 else ""

            hint_text = " ".join([t for t in [torso_hint, symmetry_hint] if t])
            if hint_text:
                cv2.putText(
                    explain,
                    hint_text,
                    (x1, min(h - 5, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return explain


def match_tracks(tracks, detections, ts):
    assignments = {}
    unmatched_det = set(range(len(detections)))
    if not tracks:
        return assignments, unmatched_det

    for track_id, track in tracks.items():
        best_idx = None
        best_dist = None
        for i, (cx, cy) in enumerate(detections):
            if i not in unmatched_det:
                continue
            dist = ((cx - track.last_cx) ** 2 + (cy - track.last_cy) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist is not None and best_dist <= TRACK_MATCH_MAX_DIST:
            assignments[track_id] = best_idx
            unmatched_det.remove(best_idx)

    return assignments, unmatched_det


def log_detections(results, scale_x, scale_y, frame_id, per_det_info, gray_frame, now_ts):
    boxes = results.boxes
    if boxes is None or boxes.xyxy is None:
        return 0, 0

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
    clses = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

    total = len(xyxy)
    if total == 0:
        return 0, 0

    log(f"frame {frame_id}: {total} detections")
    persons_in_frame = 0
    for i in range(total):
        det_index = i + 1
        x1, y1, x2, y2 = xyxy[i]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        cls_id = int(clses[i]) if len(clses) > i else -1
        conf = float(confs[i]) if len(confs) > i else 0.0
        if isinstance(model.names, dict):
            label = model.names.get(cls_id, "obj")
        else:
            label = model.names[cls_id] if 0 <= cls_id < len(model.names) else "obj"
        if label.lower() == "person":
            persons_in_frame += 1
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        track_state = None
        track_id = None
        if per_det_info and i in per_det_info:
            track_id, track_state = per_det_info[i]
        suffix = (
            f" id={track_id} state={track_state}" if track_state else ""
        )
        log(f"  {i + 1}. {label} w={width} h={height} conf={conf:.2f}{suffix}")

        if label.lower() == "person":
            mean_intensity, max_intensity = box_intensity_stats(
                gray_frame,
                (x1, y1, x2, y2),
            )
            timestamp = (
                datetime.fromtimestamp(now_ts, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            person_identifier = _format_person_id(frame_id, det_index, track_id)
            payload = {
                "timestamp": timestamp,
                "frame_id": frame_id,
                "person_id": person_identifier,
                "detection_index": det_index,
                "bbox": {"w": width, "h": height},
                "confidence": round(conf, 4),
                "motion_state": track_state,
                "temperature_mean": round(mean_intensity, 4),
                "temperature_max": round(max_intensity, 4),
            }
            log_structured(payload)

    return persons_in_frame, total


def inference_loop():
    last_processed_id = -1
    last_infer_time = 0.0
    total_person_detections = 0
    density_window = deque()
    tracks = {}
    next_track_id = 1
    while not stop_event.is_set():
        with latest_lock:
            frame_id = latest_infer["id"]
            frame = latest_infer["frame"]
            orig_frame = latest_infer["orig"]
            scale_x, scale_y = latest_infer["scale"]

        if frame is None or frame_id == last_processed_id:
            time.sleep(0.001)
            continue

        if INFER_EVERY_N > 1 and (frame_id % INFER_EVERY_N) != 0:
            last_processed_id = frame_id
            continue

        now = time.time()
        min_interval = 1.0 / MAX_INFER_FPS if MAX_INFER_FPS else 0.0
        if min_interval > 0 and now - last_infer_time < min_interval:
            time.sleep(0.001)
            continue

        results = model.predict(
            source=frame,
            imgsz=RESIZE_WIDTH or 640,
            conf=0.20,
            device="cpu",
            verbose=False,
        )

        if len(orig_frame.shape) == 2:
            gray_frame = orig_frame
        else:
            gray_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = orig_frame.shape[:2]

        boxes = results[0].boxes
        person_centers = []
        person_indices = []
        person_boxes = []
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            clses = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            for i in range(len(xyxy)):
                cls_id = int(clses[i]) if len(clses) > i else -1
                if isinstance(model.names, dict):
                    label = model.names.get(cls_id, "obj")
                else:
                    label = model.names[cls_id] if 0 <= cls_id < len(model.names) else "obj"
                if label.lower() != "person":
                    continue
                x1, y1, x2, y2 = xyxy[i]
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                person_centers.append((cx, cy))
                person_indices.append(i)
                person_boxes.append((x1, y1, x2, y2))

        now_ts = time.time()
        assignments, unmatched_det = match_tracks(tracks, person_centers, now_ts)
        for track_id, det_idx in assignments.items():
            cx, cy = person_centers[det_idx]
            tracks[track_id].update(cx, cy, now_ts)
            tracks[track_id].update_state()

            box = person_boxes[det_idx]
            mean_intensity = mean_box_intensity(gray_frame, box)
            mean_prev, std_prev = tracks[track_id].thermal_stats()
            if len(tracks[track_id].thermal_history) >= 3:
                spike_thresh = max(THERMAL_MIN_DELTA, THERMAL_SPIKE_K * std_prev)
                if mean_intensity > (mean_prev + spike_thresh):
                    if tracks[track_id].can_alert("thermal", now_ts):
                        log(
                            f"ALERT | ID-{track_id:02d} | abnormal thermal spike | possible fire proximity"
                        )
            tracks[track_id].add_thermal(now_ts, mean_intensity)

            if tracks[track_id].state == "standing":
                if (
                    tracks[track_id].standing_since is not None
                    and (now_ts - tracks[track_id].standing_since) > STANDING_ALERT_SEC
                ):
                    if tracks[track_id].can_alert("standing", now_ts):
                        log(
                            f"ALERT | ID-{track_id:02d} | motionless too long | check welfare"
                        )

        for det_idx in unmatched_det:
            cx, cy = person_centers[det_idx]
            tracks[next_track_id] = PersonTrack(next_track_id, cx, cy, now_ts)
            tracks[next_track_id].update_state()
            if not is_near_edge(cx, cy, frame_w, frame_h):
                if tracks[next_track_id].can_alert("appear", now_ts):
                    log(
                        f"ALERT | ID-{next_track_id:02d} | sudden appearance | possible occlusion"
                    )
            next_track_id += 1

        stale_ids = []
        for track_id, track in tracks.items():
            if track_id not in assignments:
                track.mark_missed()
            if track.misses > TRACK_MAX_MISSES:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            if not is_near_edge(tracks[track_id].last_cx, tracks[track_id].last_cy, frame_w, frame_h):
                if tracks[track_id].can_alert("disappear", now_ts):
                    log(
                        f"ALERT | ID-{track_id:02d} | sudden disappearance | possible occlusion"
                    )
            del tracks[track_id]

        per_det_info = {}
        for track_id, det_idx in assignments.items():
            per_det_info[person_indices[det_idx]] = (
                track_id,
                tracks[track_id].state,
            )

        persons_in_frame, total_detections = log_detections(
            results[0],
            scale_x,
            scale_y,
            frame_id,
            per_det_info,
            gray_frame,
            now_ts,
        )
        if total_detections > 0:
            total_person_detections += persons_in_frame
            density_window.append((now_ts, persons_in_frame))
            while density_window and (now_ts - density_window[0][0]) > DENSITY_WINDOW_SEC:
                density_window.popleft()
            if density_window:
                avg_persons = sum(c for _, c in density_window) / float(len(density_window))
            else:
                avg_persons = 0.0
            log(
                f"humans: frame={frame_id} in_frame={persons_in_frame} "
                f"total={total_person_detections} avg_{int(DENSITY_WINDOW_SEC)}s={avg_persons:.2f}"
            )

        color_base = apply_thermal_colormap(orig_frame)
        explain_base = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        annotated_color = draw_detections(
            color_base, results[0], scale_x, scale_y, per_det_info
        )
        annotated_explain = None
        if SHOW_EXPLAIN:
            use_cached = (
                EXPLAIN_EVERY_N > 1
                and explain_cache["frame"] is not None
                and (frame_id % EXPLAIN_EVERY_N) != 0
            )
            if use_cached:
                annotated_explain = explain_cache["frame"]
            else:
                orig_h, orig_w = orig_frame.shape[:2]
                if EXPLAIN_WIDTH and orig_w > EXPLAIN_WIDTH:
                    explain_scale = EXPLAIN_WIDTH / float(orig_w)
                    explain_w = EXPLAIN_WIDTH
                    explain_h = max(1, int(orig_h * explain_scale))
                    explain_base_small = cv2.resize(
                        explain_base, (explain_w, explain_h), interpolation=cv2.INTER_AREA
                    )
                    gray_small = cv2.resize(
                        gray_frame, (explain_w, explain_h), interpolation=cv2.INTER_AREA
                    )
                    infer_w = max(1, int(round(orig_w / float(scale_x))))
                    infer_h = max(1, int(round(orig_h / float(scale_y))))
                    scale_x_explain = explain_w / float(infer_w)
                    scale_y_explain = explain_h / float(infer_h)
                    explain_small = draw_explainability(
                        explain_base_small,
                        gray_small,
                        results[0],
                        scale_x_explain,
                        scale_y_explain,
                    )
                    annotated_explain = cv2.resize(
                        explain_small,
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    annotated_explain = draw_explainability(
                        explain_base, gray_frame, results[0], scale_x, scale_y
                    )
                explain_cache["frame"] = annotated_explain
                explain_cache["frame_id"] = frame_id

        with latest_lock:
            latest_annotated["id"] = frame_id
            latest_annotated["color"] = annotated_color
            latest_annotated["explain"] = annotated_explain

        last_processed_id = frame_id
        last_infer_time = now


def _reset_runtime():
    global latest_lock, latest_infer, latest_annotated, explain_cache, stop_event, agent_health
    latest_lock = threading.Lock()
    latest_infer = {"id": -1, "frame": None, "orig": None, "scale": (1.0, 1.0)}
    latest_annotated = {"id": -1, "color": None, "explain": None}
    explain_cache = {"frame_id": -1, "frame": None}
    stop_event = threading.Event()
    agent_health["online"] = False


def _start_agent():
    global agent_client, agent_thread, agent_health
    agent_client = setup_agent()
    agent_thread = None
    agent_health["enabled"] = AGENT_ENABLED
    agent_health["online"] = False
    if agent_client:
        agent_health["last_error"] = None
        agent_thread = threading.Thread(target=agent_watch_loop, daemon=True)
        agent_thread.start()
    else:
        agent_health["last_error"] = agent_health.get("last_error") or "AI agent unavailable."


def _capture_loop(video_path, show_windows):
    global display_delay_ms
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass
    if not cap.isOpened():
        processing_state["error"] = f"Cannot open video: {video_path}"
        stop_event.set()
        with processing_lock:
            processing_state["running"] = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
    display_delay_ms = int(1000 / fps) if fps and fps > 0 else 33
    next_frame_ts = time.perf_counter()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame
        if RESIZE_WIDTH:
            h, w = orig_frame.shape[:2]
            scale = RESIZE_WIDTH / float(w)
            infer_frame = cv2.resize(orig_frame, (RESIZE_WIDTH, int(h * scale)))
            scale_x = w / float(RESIZE_WIDTH)
            scale_y = h / float(int(h * scale))
        else:
            infer_frame = orig_frame
            scale_x, scale_y = 1.0, 1.0

        with latest_lock:
            latest_infer["id"] += 1
            latest_infer["frame"] = infer_frame
            latest_infer["orig"] = orig_frame
            latest_infer["scale"] = (scale_x, scale_y)
            annotated_color = latest_annotated["color"]
            annotated_explain = latest_annotated["explain"]

        if show_windows:
            display_color = (
                annotated_color
                if annotated_color is not None
                else apply_thermal_colormap(orig_frame)
            )
            cv2.imshow(
                WINDOW_NAME_COLOR,
                display_color if display_color is not None else orig_frame,
            )
            if SHOW_EXPLAIN:
                cv2.imshow(
                    WINDOW_NAME_EXPLAIN,
                    annotated_explain if annotated_explain is not None else display_color,
                )
            if cv2.waitKey(max(1, display_delay_ms)) & 0xFF == ord("q"):
                break

        next_frame_ts += frame_interval
        sleep_for = next_frame_ts - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)

    cap.release()
    stop_event.set()
    with processing_lock:
        processing_state["running"] = False


def start_processing(video_path, show_windows=False):
    global worker, capture_thread
    with processing_lock:
        running = processing_state["running"]
    if running:
        stop_processing()
    with processing_lock:
        processing_state["running"] = True
        processing_state["video_path"] = video_path
        processing_state["error"] = None

    reset_logging()
    _reset_runtime()
    _start_agent()
    worker = threading.Thread(target=inference_loop, daemon=True)
    worker.start()
    capture_thread = threading.Thread(
        target=_capture_loop, args=(video_path, show_windows), daemon=True
    )
    capture_thread.start()


def stop_processing():
    global worker, capture_thread, agent_thread, agent_health
    stop_event.set()
    if capture_thread:
        capture_thread.join(timeout=1.0)
    if worker:
        worker.join(timeout=1.0)
    if agent_thread:
        agent_thread.join(timeout=1.0)
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    agent_health["online"] = False
    with processing_lock:
        processing_state["running"] = False
        processing_state["video_path"] = None


def get_latest_frame(kind):
    with latest_lock:
        orig = latest_infer.get("orig")
        annotated_color = latest_annotated.get("color")
        annotated_explain = latest_annotated.get("explain")

    if orig is None:
        return None
    if kind == "color":
        return (
            annotated_color if annotated_color is not None else apply_thermal_colormap(orig)
        )
    if kind == "explain":
        if annotated_explain is not None:
            return annotated_explain
        base = (
            annotated_color if annotated_color is not None else apply_thermal_colormap(orig)
        )
        return base
    return orig


def get_latest_frame_with_id(kind):
    with latest_lock:
        frame_id = latest_infer.get("id")
        orig = latest_infer.get("orig")
        annotated_color = latest_annotated.get("color")
        annotated_explain = latest_annotated.get("explain")

    if orig is None:
        return frame_id, None
    if kind == "color":
        return (
            frame_id,
            annotated_color if annotated_color is not None else apply_thermal_colormap(orig),
        )
    if kind == "explain":
        if annotated_explain is not None:
            return frame_id, annotated_explain
        base = (
            annotated_color if annotated_color is not None else apply_thermal_colormap(orig)
        )
        return frame_id, base
    return frame_id, orig


def read_log_tail(max_lines=200):
    with log_lock:
        if log_buffer:
            return [line for _, line in list(log_buffer)[-max_lines:]]
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except Exception:
        return []
    return lines[-max_lines:]


def read_log_since(after_seq=None, max_lines=200):
    with log_lock:
        if not log_buffer:
            return [], log_seq, False
        oldest_seq = log_buffer[0][0]
        latest_seq = log_seq
        if after_seq is None:
            lines = [line for _, line in list(log_buffer)[-max_lines:]]
            return lines, latest_seq, False
        if after_seq < oldest_seq:
            lines = [line for _, line in list(log_buffer)[-max_lines:]]
            return lines, latest_seq, True
        lines = [line for seq, line in log_buffer if seq > after_seq]
        return lines, latest_seq, False


def read_latest_agent_summary(max_lines=300):
    summary = agent_health.get("last_summary")
    if summary:
        return summary
    lines = read_log_tail(max_lines=max_lines)
    for line in reversed(lines):
        if "AGENT | " in line:
            return line.split("AGENT | ", 1)[-1].strip()
    return None


def get_agent_state():
    return {
        "enabled": agent_health.get("enabled", False),
        "online": agent_health.get("online", False),
        "error": agent_health.get("last_error"),
        "last_summary_ts": agent_health.get("last_success_ts"),
    }


def read_structured_tail(max_lines=200):
    if not os.path.exists(STRUCT_LOG_PATH):
        return []
    try:
        with open(STRUCT_LOG_PATH, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except Exception:
        return []
    events = []
    for line in lines[-max_lines:]:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        normalized = _normalize_struct_event(payload)
        if not normalized:
            continue
        events.append(normalized)
    return events


def read_structured_full(person_only=True):
    if not os.path.exists(STRUCT_LOG_PATH):
        return []
    events = []
    try:
        with open(STRUCT_LOG_PATH, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                normalized = _normalize_struct_event(payload)
                if not normalized:
                    continue
                if person_only and normalized.get("person_id") in (None, "", "None"):
                    continue
                events.append(normalized)
    except Exception:
        return []
    return events


def build_activity_digest(events):
    if not events:
        return ""
    per_person = {}
    for event in events:
        person_id = event.get("person_id")
        if person_id is None:
            continue
        info = per_person.setdefault(
            person_id,
            {"count": 0, "first_frame": None, "last_frame": None, "last_event": None},
        )
        info["count"] += 1
        frame_id = event.get("frame_id")
        if info["first_frame"] is None:
            info["first_frame"] = frame_id
        info["last_frame"] = frame_id
        info["last_event"] = event
    lines = []
    for person_id in sorted(per_person):
        data = per_person[person_id]
        latest = data["last_event"] or {}
        motion = latest.get("motion_state") or "present"
        confidence = latest.get("confidence")
        confidence_str = "unknown" if confidence is None else f"{confidence:.2f}"
        timestamp = latest.get("timestamp") or "unknown time"
        span = data["first_frame"]
        if data["last_frame"] != data["first_frame"]:
            span = f"{data['first_frame']}->{data['last_frame']}"
        lines.append(
            (
                f"Person {person_id}: {data['count']} entries, last seen {motion} in frame "
                f"{latest.get('frame_id')} (frames {span}), confidence {confidence_str} at {timestamp}."
            )
        )
    return "\n".join(lines)


def summarize_recent_people(events, limit=4):
    if not events:
        return []
    latest_by_person = {}
    for event in events:
        person_id = event.get("person_id")
        if person_id is None:
            continue
        latest_by_person[person_id] = event
    sorted_events = sorted(
        latest_by_person.values(),
        key=lambda item: item.get("timestamp", ""),
        reverse=True,
    )
    summary = []
    for event in sorted_events[:limit]:
        person_id = event.get("person_id")
        frame_id = event.get("frame_id")
        motion_state = event.get("motion_state") or "present"
        confidence = event.get("confidence")
        timestamp = event.get("timestamp")
        confidence_str = "unknown" if confidence is None else f"{confidence:.2f}"
        summary_text = (
            f"Human activity detected: Person {person_id} was {motion_state} "
            f"in frame {frame_id} (confidence {confidence_str}) at {timestamp}."
        )
        summary.append(
            {
                "person_id": person_id,
                "frame_id": frame_id,
                "motion_state": motion_state,
                "confidence": confidence,
                "timestamp": timestamp,
                "summary": summary_text,
            }
        )
    return summary


def run_video(video_path):
    log("Model loaded.")
    log(f"Testing on: {video_path}")
    start_processing(video_path, show_windows=True)
    while not stop_event.is_set():
        time.sleep(0.1)
    stop_processing()


if __name__ == "__main__":
    sample_video = os.path.join(
        PROJECT_ROOT,
        "images",
        "WhatsApp Video 2026-02-12 at 12.58.35 PM.mp4",
    )
    run_video(sample_video)






















































