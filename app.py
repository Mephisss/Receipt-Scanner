import io
import json
import logging
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from model_llm import LLMReceiptExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
try:
    extractor = LLMReceiptExtractor()
    logger.info("LLM extractor initialized successfully")
except ValueError as e:
    logger.error(f"Failed to initialize extractor: {e}")
    logger.error("Make sure GROQ_API_KEY is set in environment variables")
    extractor = None


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}

# Model is loaded once at startup and reused across requests
extractor = LLMReceiptExtractor()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        logger.info(f"Analysing image: {file.filename} ({len(image_bytes) // 1024} KB, {image.size})")

        t0 = time.perf_counter()
        result = extractor.extract(image)
        elapsed = time.perf_counter() - t0

        result["processing_time_s"] = round(elapsed, 2)
        logger.info(f"Done in {elapsed:.2f}s — {len(result['items'])} items, total={result['total']}")

        return jsonify(result)

    except Exception as e:
        logger.exception("Extraction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    """Health check."""
    return jsonify({
        "status": "ok",
        "model": extractor.model if extractor else "not initialized",
        "api_configured": extractor is not None,
    })


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Receipt scanner ready — starting on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
