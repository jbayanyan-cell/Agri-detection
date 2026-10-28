"""Shared ONNX pest detection engine for Vercel (YOLOv11 rice-pests model)."""

from __future__ import annotations

import io
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None  # type: ignore

# Exact order from Roboflow test-4fdzn/rice-pests-ajory-1-yolo11n-t1 (15 logits)
CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "black bug",
    "brown plant hopper",
    "green leaf hopper",
    "object",
    "rice bug",
    "rice grasshopper",
    "rice leaf roller",
    "stem borer",
]

PESTICIDE_RECS = {
    "black bug": "Carbaryl dust or fipronil bait at tillering; keep field sanitation.",
    "brown plant hopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
    "green leaf hopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance.",
    "rice bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
    "rice grasshopper": "Handpick when few; use carbaryl or lambda-cyhalothrin if heavy; protect field edges.",
    "rice leaf roller": "Cartap or chlorantraniliprole early; avoid late unnecessary sprays.",
    "stem borer": "Use chlorantraniliprole or cartap early; remove stubbles after harvest.",
    "object": "Unclassified detection — verify in the field before treating.",
}

# Numeric Roboflow duplicates → real pest names (same insects, two label systems)
CLASS_ALIASES = {
    "0": "black bug",
    "1": "brown plant hopper",
    "2": "green leaf hopper",
    "3": "rice bug",
    "4": "rice grasshopper",
    "5": "rice leaf roller",
    "6": "stem borer",
}

# Still ignored after aliasing (not a real pest class)
DISABLED_CLASSES = frozenset({
    "object",
})


def _is_disabled_class(name: str) -> bool:
    return name.strip().lower() in DISABLED_CLASSES


def _canonicalize_class_name(name: str) -> str:
    """Map duplicate numeric labels to canonical pest names."""
    key = name.strip().lower()
    return CLASS_ALIASES.get(key, name.strip())

_session = None
_input_details = None
_output_details = None
_model_path: Optional[str] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_class_names_file() -> Optional[List[str]]:
    path = _project_root() / "models" / "class_names.txt"
    if not path.exists():
        return None
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        names.append(name)
    return names or None


def get_class_names() -> List[str]:
    return _load_class_names_file() or list(CLASS_NAMES)


def get_reportable_class_names() -> List[str]:
    """Canonical pest labels for health/counts (digits aliased; junk excluded)."""
    seen = set()
    names: List[str] = []
    for raw in get_class_names():
        if not raw:
            continue
        name = _canonicalize_class_name(raw)
        if _is_disabled_class(name) or name.isdigit():
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _model_candidates() -> list[Path]:
    root = _project_root()
    # Prefer the new YOLOv11 Roboflow export
    names = [
        "weights.onnx",
        "yolov5s_weights.onnx",
        "best.onnx",
        "best 2.onnx",
        "best5.onnx",
    ]
    candidates: list[Path] = []
    for name in names:
        candidates.append(root / "models" / name)
    for name in names:
        candidates.append(root / "deployment" / "models" / name)
    return candidates


def _download_model(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def find_onnx_model() -> str:
    for candidate in _model_candidates():
        if candidate.exists():
            return str(candidate)

    model_url = os.getenv("MODEL_URL", "").strip()
    if model_url:
        cache_dir = Path("/tmp/agrishield_models")
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / "model.onnx"
        ttl = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "0") or "0")
        need_download = True
        if dest.exists() and ttl > 0:
            age = time.time() - dest.stat().st_mtime
            need_download = age > ttl
        if need_download:
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            _download_model(model_url, dest)
        return str(dest)

    raise FileNotFoundError(
        "ONNX model not found. Add models/weights.onnx or set MODEL_URL."
    )


def load_onnx_model(model_path: str):
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_details = sess.get_inputs()[0]
    output_details = sess.get_outputs()[0]
    return sess, input_details, output_details


def get_model() -> Tuple[Any, Any, Any, Optional[str]]:
    global _session, _input_details, _output_details, _model_path

    if _session is not None:
        return _session, _input_details, _output_details, _model_path

    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime not available")

    _model_path = find_onnx_model()
    _session, _input_details, _output_details = load_onnx_model(_model_path)
    return _session, _input_details, _output_details, _model_path


def preprocess_image(image: Image.Image, input_shape: tuple) -> np.ndarray:
    """Stretch resize (matches Roboflow 'Stretch to')."""
    if len(input_shape) == 4:
        if input_shape[1] == 3:
            h, w = int(input_shape[2]), int(input_shape[3])
        else:
            h, w = int(input_shape[1]), int(input_shape[2])
    else:
        h, w = 640, 640

    img = image.resize((w, h), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0

    if len(img_array.shape) == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

    return img_array


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_per_class(
    candidates: List[Dict[str, Any]],
    iou_threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_class.setdefault(c["class"], []).append(c)

    for _, items in by_class.items():
        items = sorted(items, key=lambda d: d["confidence"], reverse=True)
        selected: List[Dict[str, Any]] = []
        while items:
            best = items.pop(0)
            selected.append(best)
            items = [
                other
                for other in items
                if _box_iou(best["xyxy"], other["xyxy"]) < iou_threshold
            ]
        kept.extend(selected)

    kept.sort(key=lambda d: d["confidence"], reverse=True)
    return kept


def _normalize_detections(output_data: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Return (detections [N, C], channel_first).

    YOLOv8/v11 Ultralytics/Roboflow: [1, 4+nc, N]  ΓåÆ transpose to [N, 4+nc]
    YOLOv5 style:                    [1, N, 5+nc]  ΓåÆ [N, 5+nc]
    """
    if output_data.ndim == 3:
        arr = output_data[0]
    elif output_data.ndim == 2:
        arr = output_data
    else:
        return np.zeros((0, 6), dtype=np.float32), False

    channel_first = arr.shape[0] < arr.shape[1]
    if channel_first:
        arr = arr.T
    return arr, channel_first


def _extract_candidates(
    output_data: np.ndarray,
    conf_threshold: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Decode YOLO outputs into candidates in original-image pixel space.
    Supports YOLOv11 [1, 4+nc, N] and YOLOv5 [1, N, 5+nc].
    """
    class_names = get_class_names()
    candidates: List[Dict[str, Any]] = []

    detections, channel_first = _normalize_detections(output_data)
    if detections.size == 0:
        return candidates

    channels = int(detections.shape[1])
    if channel_first:
        has_objectness = False
        model_nc = max(1, channels - 4)
    else:
        has_objectness = True
        model_nc = max(1, channels - 5)

    score_start = 5 if has_objectness else 4

    for det in detections:
        if det.shape[0] < score_start + 1:
            continue

        cx, cy, bw, bh = float(det[0]), float(det[1]), float(det[2]), float(det[3])
        class_scores = det[score_start : score_start + model_nc]
        if class_scores.size == 0:
            continue

        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        conf = float(det[4]) * class_conf if has_objectness else class_conf
        if conf < conf_threshold:
            continue

        if 0 <= class_id < len(class_names):
            name = class_names[class_id]
        else:
            name = f"class_{class_id}"

        name = _canonicalize_class_name(name)
        if _is_disabled_class(name) or name.isdigit():
            continue

        # Map from model-input coords → crop/original pixel coords
        cx = cx * scale_x + offset_x
        cy = cy * scale_y + offset_y
        bw = bw * scale_x
        bh = bh * scale_y
        candidates.append(
            {
                "class": name,
                "confidence": conf,
                "x": cx,
                "y": cy,
                "width": bw,
                "height": bh,
                "xyxy": _xywh_to_xyxy(cx, cy, bw, bh),
            }
        )

    return candidates


def _model_hw(input_details) -> Tuple[int, int]:
    shape = input_details.shape if input_details.shape else [1, 3, 640, 640]
    if len(shape) == 4:
        if shape[1] == 3:
            return int(shape[2]), int(shape[3])
        return int(shape[1]), int(shape[2])
    return 640, 640


def _infer_crop(
    session,
    input_details,
    output_details,
    crop: Image.Image,
    conf_threshold: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run one forward pass on a crop; return boxes in full-image coords."""
    mh, mw = _model_hw(input_details)
    cw, ch = crop.size
    if cw < 1 or ch < 1:
        return []
    input_shape = input_details.shape if input_details.shape else [1, 3, mh, mw]
    input_data = preprocess_image(crop, input_shape)
    output = session.run([output_details.name], {input_details.name: input_data})
    return _extract_candidates(
        output[0],
        conf_threshold=conf_threshold,
        scale_x=cw / float(mw),
        scale_y=ch / float(mh),
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _tile_windows(
    width: int,
    height: int,
    grid: int,
    overlap: float,
) -> List[Tuple[int, int, int, int]]:
    """Axis-aligned overlapping tiles as (x1, y1, x2, y2)."""
    grid = max(1, grid)
    overlap = min(0.5, max(0.0, overlap))
    if grid == 1:
        return [(0, 0, width, height)]

    windows: List[Tuple[int, int, int, int]] = []
    step_x = width / grid
    step_y = height / grid
    tile_w = int(round(step_x * (1.0 + overlap)))
    tile_h = int(round(step_y * (1.0 + overlap)))
    tile_w = min(width, max(tile_w, int(round(step_x))))
    tile_h = min(height, max(tile_h, int(round(step_y))))

    for gy in range(grid):
        for gx in range(grid):
            x1 = int(round(gx * step_x))
            y1 = int(round(gy * step_y))
            if gx == grid - 1:
                x1 = max(0, width - tile_w)
            if gy == grid - 1:
                y1 = max(0, height - tile_h)
            x2 = min(width, x1 + tile_w)
            y2 = min(height, y1 + tile_h)
            windows.append((x1, y1, x2, y2))
    return windows


def _candidates_to_counts_and_details(
    kept: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    report_names = get_reportable_class_names()
    counts = {name: 0 for name in report_names}
    details: List[Dict[str, Any]] = []
    for item in kept:
        name = item["class"]
        counts[name] = counts.get(name, 0) + 1
        details.append(
            {
                "class": name,
                "confidence": round(float(item["confidence"]), 4),
                "x": float(item["x"]),
                "y": float(item["y"]),
                "width": float(item["width"]),
                "height": float(item["height"]),
            }
        )
    return counts, details


def postprocess_yolo(
    output_data: np.ndarray,
    conf_threshold: float = 0.10,
    iou_threshold: float = 0.20,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Single-pass decode + NMS (kept for compatibility)."""
    candidates = _extract_candidates(output_data, conf_threshold=conf_threshold)
    kept = _nms_per_class(candidates, iou_threshold=iou_threshold)
    return _candidates_to_counts_and_details(kept)


def run_detection(image_bytes: bytes) -> Dict[str, Any]:
    session, input_details, output_details, model_path = get_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t0 = time.time()

    # Dense swarms: low conf + soft NMS + overlapping tiles so small bugs survive resize.
    conf = float(os.getenv("DETECT_CONFIDENCE", "0.08") or "0.08")
    iou = float(os.getenv("DETECT_IOU", "0.20") or "0.20")
    max_preds = int(os.getenv("DETECT_MAX_PREDICTIONS", "500") or "500")
    use_tiles = (os.getenv("DETECT_TILES", "1") or "1").strip().lower() not in {
        "0", "false", "off", "no",
    }
    tile_grid = int(os.getenv("DETECT_TILE_GRID", "2") or "2")
    tile_overlap = float(os.getenv("DETECT_TILE_OVERLAP", "0.30") or "0.30")
    if max_preds < 1:
        max_preds = 500

    orig_w, orig_h = img.size
    all_candidates: List[Dict[str, Any]] = []

    # Full-frame pass
    all_candidates.extend(
        _infer_crop(session, input_details, output_details, img, conf)
    )

    # Tile passes — critical for packed black-bug clusters
    tile_runs = 0
    if use_tiles and tile_grid > 1 and orig_w >= 320 and orig_h >= 320:
        for x1, y1, x2, y2 in _tile_windows(orig_w, orig_h, tile_grid, tile_overlap):
            crop = img.crop((x1, y1, x2, y2))
            all_candidates.extend(
                _infer_crop(
                    session,
                    input_details,
                    output_details,
                    crop,
                    conf,
                    offset_x=float(x1),
                    offset_y=float(y1),
                )
            )
            tile_runs += 1

    kept = _nms_per_class(all_candidates, iou_threshold=iou)
    counts, predictions = _candidates_to_counts_and_details(kept)

    recommendations = {k: v for k, v in PESTICIDE_RECS.items() if counts.get(k, 0) > 0}
    total = sum(counts.values())

    dominant = "no_pest_detected"
    dominant_conf = 0.0
    common_name = "No Pest Detected"
    if total > 0:
        dominant = max(counts.items(), key=lambda kv: kv[1])[0]
        class_preds = [p for p in predictions if p["class"] == dominant]
        dominant_conf = max((p["confidence"] for p in class_preds), default=0.0)
        common_name = dominant.title()

    return {
        "status": "success",
        "pest_counts": counts,
        "recommendations": recommendations,
        "total_pests_detected": total,
        "predictions": predictions[:max_preds],
        "detections": predictions[:max_preds],
        "predicted_class": dominant,
        "confidence": round(float(dominant_conf), 4),
        "common_name": common_name,
        "inference_time_ms": round((time.time() - t0) * 1000, 1),
        "model": Path(model_path).name if model_path else "none",
        "framework": "ONNX Runtime (YOLOv11)",
        "backend": "onnx",
        "count_method": "tiled_per_class_nms" if tile_runs else "per_class_nms",
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "max_predictions": max_preds,
        "tile_grid": tile_grid if tile_runs else 1,
        "tile_runs": tile_runs,
    }


def health_payload() -> Dict[str, Any]:
    if not ONNX_AVAILABLE:
        return {
            "status": "error",
            "message": "ONNX Runtime not available",
            "install": "pip install onnxruntime",
        }

    try:
        _, input_details, _, model_path = get_model()
        names = get_class_names()
        reportable = get_reportable_class_names()
        return {
            "status": "ok",
            "model": Path(model_path).name if model_path else "none",
            "input_shape": list(input_details.shape) if input_details is not None else None,
            "classes": reportable,
            "class_aliases": dict(CLASS_ALIASES),
            "disabled_classes": sorted(DISABLED_CLASSES),
            "num_classes": len(names),
            "framework": "ONNX Runtime (YOLOv11)",
            "backend": "onnx",
            "platform": "vercel",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
