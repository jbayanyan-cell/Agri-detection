"""Shared ONNX pest detection engine for Vercel (YOLOv5 rice-pests model)."""

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

# Order must match Roboflow common-rice-pests-philippines/11 class_names.txt
CLASS_NAMES = [
    "brown-planthopper",
    "green-leafhopper",
    "leaf-folder",
    "rice-bug",
    "stem-borer",
    "whorl-maggot",
]

PESTICIDE_RECS = {
    "brown-planthopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
    "green-leafhopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance.",
    "leaf-folder": "Use cartap or chlorantraniliprole when leaf damage is rising; conserve natural enemies.",
    "rice-bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
    "stem-borer": "Use chlorantraniliprole or cartap early; remove stubbles after harvest.",
    "whorl-maggot": "Apply early protective insecticide if damage is severe; improve field drainage.",
}

_session = None
_input_details = None
_output_details = None
_model_path: Optional[str] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _model_candidates() -> list[Path]:
    root = _project_root()
    # Prefer the bundled Roboflow YOLOv5 export
    names = [
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
        "ONNX model not found. Add models/yolov5s_weights.onnx or set MODEL_URL."
    )


def load_onnx_model(model_path: str):
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_details = sess.get_inputs()[0]
    # YOLOv5 exports several outputs; use the concatenated detections tensor
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
    """Letterbox-free stretch resize (matches Roboflow 'Stretch to' 416x416)."""
    if len(input_shape) == 4:
        if input_shape[1] == 3:
            h, w = int(input_shape[2]), int(input_shape[3])
        else:
            h, w = int(input_shape[1]), int(input_shape[2])
    else:
        h, w = 416, 416

    img = image.resize((w, h), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0

    if len(img_array.shape) == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

    return img_array


def postprocess_yolov5(
    output_data: np.ndarray,
    conf_threshold: float = 0.25,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    YOLOv5 ONNX output: [1, N, 5+num_classes]
    columns = x, y, w, h, objectness, class_scores...
    """
    counts = {name: 0 for name in CLASS_NAMES}
    details: List[Dict[str, Any]] = []

    if output_data.ndim == 3:
        detections = output_data[0]
    elif output_data.ndim == 2:
        detections = output_data
    else:
        return counts, details

    num_classes = len(CLASS_NAMES)
    for det in detections:
        if det.shape[0] < 5 + num_classes:
            # fallback: old [x,y,w,h,conf,class_id]
            if det.shape[0] >= 6:
                conf = float(det[4])
                class_id = int(det[5])
                if conf >= conf_threshold and 0 <= class_id < num_classes:
                    name = CLASS_NAMES[class_id]
                    counts[name] += 1
                    details.append(
                        {
                            "class": name,
                            "confidence": round(conf, 4),
                            "x": float(det[0]),
                            "y": float(det[1]),
                            "width": float(det[2]),
                            "height": float(det[3]),
                        }
                    )
            continue

        objectness = float(det[4])
        class_scores = det[5 : 5 + num_classes]
        class_id = int(np.argmax(class_scores))
        class_conf = float(class_scores[class_id])
        conf = objectness * class_conf
        if conf < conf_threshold:
            continue
        if not (0 <= class_id < num_classes):
            continue

        name = CLASS_NAMES[class_id]
        counts[name] += 1
        details.append(
            {
                "class": name,
                "confidence": round(conf, 4),
                "x": float(det[0]),
                "y": float(det[1]),
                "width": float(det[2]),
                "height": float(det[3]),
            }
        )

    # Keep highest-confidence boxes first
    details.sort(key=lambda d: d["confidence"], reverse=True)
    return counts, details


def run_detection(image_bytes: bytes) -> Dict[str, Any]:
    session, input_details, output_details, model_path = get_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t0 = time.time()

    input_shape = input_details.shape if input_details.shape else [1, 3, 416, 416]
    input_data = preprocess_image(img, input_shape)
    output = session.run([output_details.name], {input_details.name: input_data})

    conf = float(os.getenv("DETECT_CONFIDENCE", "0.25") or "0.25")
    counts, predictions = postprocess_yolov5(output[0], conf_threshold=conf)

    recommendations = {k: v for k, v in PESTICIDE_RECS.items() if counts.get(k, 0) > 0}
    total = sum(counts.values())

    return {
        "status": "success",
        "pest_counts": counts,
        "recommendations": recommendations,
        "total_pests_detected": total,
        "predictions": predictions[:50],
        "inference_time_ms": round((time.time() - t0) * 1000, 1),
        "model": Path(model_path).name if model_path else "none",
        "framework": "ONNX Runtime (YOLOv5)",
        "backend": "onnx",
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
        return {
            "status": "ok",
            "model": Path(model_path).name if model_path else "none",
            "input_shape": list(input_details.shape) if input_details is not None else None,
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "framework": "ONNX Runtime (YOLOv5)",
            "backend": "onnx",
            "platform": "vercel",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
