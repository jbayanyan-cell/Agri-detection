"""Shared ONNX pest detection engine for Vercel serverless."""

from __future__ import annotations

import io
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None  # type: ignore

CLASS_NAMES = [
    "Rice_Bug",
    "White stem borer",
    "black-bug",
    "brown_hopper",
    "green_hopper",
]

PESTICIDE_RECS = {
    "Rice_Bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
    "green_hopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance.",
    "brown_hopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
    "black-bug": "Carbaryl dust or fipronil bait at tillering; field sanitation recommended.",
}

_session = None
_input_details = None
_output_details = None
_model_path: Optional[str] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _model_candidates() -> list[Path]:
    root = _project_root()
    names = ["best 2.onnx", "best.onnx", "best5.onnx"]
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
        # Refresh from PHP active-model URL on each cold start so Deploy updates take effect.
        # Set MODEL_CACHE_TTL_SECONDS (default 0 = always refresh) to keep a short cache.
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
        "ONNX model not found. Add models/best.onnx or set MODEL_URL env var."
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
    if len(input_shape) == 4:
        if input_shape[1] == 3:
            h, w = input_shape[2], input_shape[3]
        else:
            h, w = input_shape[1], input_shape[2]
    else:
        h, w = 512, 512

    img = image.resize((w, h))
    img_array = np.array(img, dtype=np.float32) / 255.0

    if len(img_array.shape) == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

    return img_array


def postprocess_output(output_data: np.ndarray, conf_threshold: float = 0.15) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}

    if len(output_data.shape) == 3:
        detections = output_data[0]
    elif len(output_data.shape) == 2:
        detections = output_data
    else:
        return counts

    for detection in detections:
        if len(detection) >= 6:
            conf = float(detection[4])
            class_id = int(detection[5])
            if conf >= conf_threshold and 0 <= class_id < len(CLASS_NAMES):
                counts[CLASS_NAMES[class_id]] += 1

    return counts


def run_detection(image_bytes: bytes) -> Dict[str, Any]:
    session, input_details, output_details, model_path = get_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t0 = time.time()

    input_shape = input_details.shape if input_details.shape else [1, 3, 512, 512]
    input_data = preprocess_image(img, input_shape)
    output = session.run([output_details.name], {input_details.name: input_data})
    counts = postprocess_output(output[0], conf_threshold=0.15)

    recommendations = {k: v for k, v in PESTICIDE_RECS.items() if counts.get(k, 0) > 0}
    total = sum(counts.values())

    return {
        "status": "success",
        "pest_counts": counts,
        "recommendations": recommendations,
        "total_pests_detected": total,
        "inference_time_ms": round((time.time() - t0) * 1000, 1),
        "model": Path(model_path).name if model_path else "none",
        "framework": "ONNX Runtime",
    }


def health_payload() -> Dict[str, Any]:
    if not ONNX_AVAILABLE:
        return {
            "status": "error",
            "message": "ONNX Runtime not available",
            "install": "pip install onnxruntime",
        }

    try:
        _, _, _, model_path = get_model()
        return {
            "status": "ok",
            "model": Path(model_path).name if model_path else "none",
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "framework": "ONNX Runtime",
            "platform": "vercel",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
