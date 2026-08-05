"""AgriShield ML API — Vercel serverless (detect + health)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from vercel/lib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request

from lib.ml_engine import health_payload, run_detection

app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/", methods=["GET"])
def root():
    health = health_payload()
    return jsonify({
        "name": "AgriShield Pest Detection API",
        "version": "1.1.0",
        "platform": "vercel",
        "framework": health.get("framework", "ONNX Runtime (YOLOv5)"),
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
        },
        "model_loaded": health.get("status") == "ok",
        "model": health.get("model", "none"),
    })


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return "", 204
    payload = health_payload()
    code = 200 if payload.get("status") == "ok" else 500
    return jsonify(payload), code


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 204

    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "empty image data"}), 400
        result = run_detection(image_bytes)
        return jsonify(result)
    except RuntimeError as exc:
        return jsonify({"error": "ONNX model not available", "message": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": "detection failed", "message": str(exc)}), 500


# Vercel expects the Flask app instance
app = app
