import os
import atexit
import shutil
import threading
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

BBOX_COLOR = (255, 255, 255)
BBOX_THICKNESS = 3
TEXT_SCALE = 1.0
TEXT_THICKNESS = 3
DASH_LEN = 6
DASH_GAP = 4
CAM_BOX_THICKNESS = 2
CAM_TEXT_SCALE = 0.9
CAM_TEXT_THICKNESS = 2
TITLE_FONT_SIZE = 18

MODEL_REPO_ID = "say89/PHOENIXV9_FIRE_MV"
MODEL_FILE = "FIREDETCT009M671.pt"
TEST_IMAGE_CANDIDATES = [
    r"Project\images\Fire_test.png",
    r"images\Fire_test.png",
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".model_cache")
try:
    os.makedirs(CACHE_ROOT, exist_ok=True)
    RUN_CACHE_DIR = tempfile.mkdtemp(prefix="hf_run_", dir=CACHE_ROOT)
except Exception:
    RUN_CACHE_DIR = tempfile.mkdtemp(prefix="hf_run_")


def _cleanup_model_cache():
    try:
        shutil.rmtree(RUN_CACHE_DIR, ignore_errors=True)
    except Exception:
        pass


atexit.register(_cleanup_model_cache)

_model_lock = threading.Lock()
_model_instance = None


def preprocess_thermal_image_safe(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    # Handle 16-bit thermal.
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # Handle float images.
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # Grayscale -> BGR.
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Final safety check.
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape after preprocessing: {img.shape}")

    return img


def create_gradcam_overlay(base_bgr, result, model, alpha=0.45):
    overlay = base_bgr.copy()
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return overlay, False

    xyxy = boxes.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()

    classes = boxes.cls if getattr(boxes, "cls", None) is not None else None
    if hasattr(classes, "cpu"):
        classes = classes.cpu().numpy()

    edges = cv2.Canny(gray, 50, 120)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    h, w = base_bgr.shape[:2]
    has_overlay = False

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        heat = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        roi_base = overlay[y1:y2, x1:x2]
        overlay[y1:y2, x1:x2] = cv2.addWeighted(roi_base, 1.0 - alpha, heat_color, alpha, 0)

        edge_roi = edges_color[y1:y2, x1:x2]
        overlay[y1:y2, x1:x2] = cv2.addWeighted(overlay[y1:y2, x1:x2], 0.85, edge_roi, 0.15, 0)

        cls_id = int(classes[i]) if classes is not None and i < len(classes) else -1
        if isinstance(model.names, dict):
            label = model.names.get(cls_id, "obj")
        else:
            label = model.names[cls_id] if 0 <= cls_id < len(model.names) else "obj"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), BBOX_COLOR, CAM_BOX_THICKNESS)
        cv2.putText(
            overlay,
            f"{label} CAM",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            CAM_TEXT_SCALE,
            BBOX_COLOR,
            CAM_TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        has_overlay = True

    return overlay, has_overlay


def draw_detections(base_frame, result, model, scale_x=1.0, scale_y=1.0):
    annotated = base_frame.copy() if base_frame is not None else None
    if annotated is None:
        return base_frame

    boxes = result.boxes
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
        text = f"{label} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

        cy = (y1 + y2) // 2
        x = x1
        while x < x2:
            x_end = min(x + DASH_LEN, x2)
            cv2.line(annotated, (x, cy), (x_end, cy), BBOX_COLOR, BBOX_THICKNESS)
            x += DASH_LEN + DASH_GAP

        bar_w = max(40, min(100, bw))
        bar_h = 4
        bar_x1 = x1
        bar_y1 = min(annotated.shape[0] - bar_h - 1, y1 + 2)
        bar_x2 = bar_x1 + bar_w
        bar_y2 = bar_y1 + bar_h
        fill_w = int(bar_w * max(0.0, min(1.0, conf)))
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 2)
        if fill_w > 0:
            cv2.rectangle(
                annotated,
                (bar_x1, bar_y1),
                (bar_x1 + fill_w, bar_y2),
                BBOX_COLOR,
                -1,
            )

        cv2.putText(
            annotated,
            text,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            BBOX_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated


def resolve_test_image_path():
    for candidate in TEST_IMAGE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Test image not found. Tried: " + ", ".join(TEST_IMAGE_CANDIDATES)
    )


def run_test():
    model = get_model()

    img_path = resolve_test_image_path()
    img = preprocess_thermal_image_safe(img_path)
    results = model.predict(source=[img], conf=0.25, verbose=False)
    result = results[0]

    annotated = draw_detections(img, result, model)
    cam_overlay_bgr, has_cam = create_gradcam_overlay(img, result, model)
    cam_overlay_rgb = cv2.cvtColor(cam_overlay_bgr, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(annotated_rgb)
    axes[0].set_title("Detection Prediction", fontsize=TITLE_FONT_SIZE)
    axes[0].axis("off")

    axes[1].imshow(cam_overlay_rgb)
    if has_cam:
        axes[1].set_title("Grad-CAM Overlay Prediction", fontsize=TITLE_FONT_SIZE)
    else:
        axes[1].set_title(
            "Grad-CAM Overlay Prediction (No detections returned)",
            fontsize=TITLE_FONT_SIZE,
        )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def get_model():
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    with _model_lock:
        if _model_instance is not None:
            return _model_instance
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILE,
            repo_type="space",
            cache_dir=RUN_CACHE_DIR,
        )
        _model_instance = YOLO(model_path)
        return _model_instance


def detect_from_image_path(img_path, conf=0.25):
    model = get_model()
    img = preprocess_thermal_image_safe(img_path)
    results = model.predict(source=[img], conf=conf, verbose=False)
    result = results[0]

    annotated = draw_detections(img, result, model)
    cam_overlay_bgr, has_cam = create_gradcam_overlay(img, result, model)

    boxes = []
    if result.boxes is not None and result.boxes.xyxy is not None:
        xyxy = result.boxes.xyxy
        confs = result.boxes.conf
        clses = result.boxes.cls
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        if hasattr(confs, "cpu"):
            confs = confs.cpu().numpy()
        if hasattr(clses, "cpu"):
            clses = clses.cpu().numpy()
        for i in range(len(xyxy)):
            cls_id = int(clses[i]) if len(clses) > i else -1
            if isinstance(model.names, dict):
                label = model.names.get(cls_id, "obj")
            else:
                label = model.names[cls_id] if 0 <= cls_id < len(model.names) else "obj"
            boxes.append(
                {
                    "class": cls_id,
                    "label": label,
                    "confidence": float(confs[i]) if len(confs) > i else 0.0,
                    "bbox": xyxy[i].tolist(),
                }
            )

    return {
        "boxes": boxes,
        "annotated_bgr": annotated,
        "cam_overlay_bgr": cam_overlay_bgr,
        "has_cam": has_cam,
    }


if __name__ == "__main__":
    run_test()





