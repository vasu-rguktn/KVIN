"""
api.py  –  FastAPI backend for the Fracture Detection pipeline
Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoint:
    POST /predict
        Accepts:  multipart/form-data  →  file: image bytes
        Returns:  JSON prediction result
"""

import io
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Configuration  – adjust paths to match your project layout
# ─────────────────────────────────────────────────────────────────
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

BODY_PART_MODEL_PATH = os.path.join(THIS_FOLDER, "weights", "ResNet50_BodyParts_unfreeze_aug.keras")
FRACTURE_MODEL_PATHS = {
    "Elbow":    os.path.join(THIS_FOLDER, "weights", "ResNet50_Elbow_frac_unfreeze_aug.keras"),
    "Hand":     os.path.join(THIS_FOLDER, "weights", "ResNet50_Hand_frac_unfreeze_aug.keras"),
    "Shoulder": os.path.join(THIS_FOLDER, "weights", "ResNet50_Shoulder_frac_unfreeze_aug.keras"),
}

# Must match the alphabetical order LabelEncoder used during training
BODY_PART_CLASSES = ["Elbow", "Hand", "Shoulder"]

IMG_SIZE = 224
FRACTURE_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────
# Global model registry  (loaded once at startup via lifespan)
# ─────────────────────────────────────────────────────────────────
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all Keras models once when the server starts."""
    log.info("Loading body-part identification model …")
    models["body_part"] = tf.keras.models.load_model(BODY_PART_MODEL_PATH)
    log.info("  ✓ body-part model ready")

    models["fracture"] = {}
    for part, path in FRACTURE_MODEL_PATHS.items():
        log.info(f"Loading fracture model for {part} …")
        models["fracture"][part] = tf.keras.models.load_model(path)
        log.info(f"  ✓ {part} fracture model ready")

    log.info("All models loaded – API is ready.")
    yield
    # Nothing to clean up, but you could call model.release() here if needed.


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fracture Detection API",
    description="Two-phase pipeline: body-part identification → fracture detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    filename: str

    # Phase 1
    predicted_part: str
    part_confidence: float            # 0-1, confidence for the predicted class
    part_probabilities: dict          # {part: probability} for all classes

    # Phase 2
    predicted_fracture: str           # "fractured" | "normal"
    fracture_confidence: float        # raw sigmoid output (probability of fracture)
    fracture_label: str               # human-friendly label

    # Convenience flag
    is_fractured: bool


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


# ─────────────────────────────────────────────────────────────────
# Preprocessing  (identical to training)
# ─────────────────────────────────────────────────────────────────
def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes → preprocessed numpy array ready for ResNet50."""
    tensor = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    tensor = tf.image.resize(tensor, (IMG_SIZE, IMG_SIZE))
    tensor = tf.keras.applications.resnet50.preprocess_input(tensor)
    return tensor.numpy()


# ─────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────
def run_body_part_inference(image: np.ndarray) -> tuple[str, float, dict]:
    inp   = np.expand_dims(image, axis=0)
    probs = models["body_part"].predict(inp, verbose=0)[0]          # shape: (3,)
    idx   = int(np.argmax(probs))
    predicted_part = BODY_PART_CLASSES[idx]
    confidence     = float(probs[idx])
    prob_dict      = {cls: round(float(p), 4) for cls, p in zip(BODY_PART_CLASSES, probs)}
    return predicted_part, confidence, prob_dict


def run_fracture_inference(part: str, image: np.ndarray) -> tuple[str, float]:
    frac_model = models["fracture"].get(part)
    if frac_model is None:
        raise ValueError(f"No fracture model registered for part: {part}")

    inp  = np.expand_dims(image, axis=0)
    prob = float(frac_model.predict(inp, verbose=0)[0][0])          # sigmoid output
    label = "fractured" if prob >= FRACTURE_THRESHOLD else "normal"
    return label, prob


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check():
    """Quick liveness check – also reports which models are in memory."""
    loaded = ["body_part"] + list(models.get("fracture", {}).keys())
    return HealthResponse(status="ok", models_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="X-ray image (JPEG / PNG / WEBP)")):
    """
    Two-phase fracture detection on a single uploaded X-ray image.

    1. Identifies the body part (Elbow / Hand / Shoulder).
    2. Runs the matching fracture-detection model.
    Returns structured JSON with all confidence scores.
    """
    # ── validate content type ──────────────────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG, PNG, or WEBP image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # ── preprocess ────────────────────────────────────────────────
    try:
        image = preprocess_bytes(image_bytes)
    except Exception as e:
        log.exception("Preprocessing failed")
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    # ── Phase 1: body-part identification ─────────────────────────
    try:
        pred_part, part_conf, part_probs = run_body_part_inference(image)
    except Exception as e:
        log.exception("Body-part inference failed")
        raise HTTPException(status_code=500, detail=f"Body-part model error: {e}")

    log.info(f"[{file.filename}] Predicted part: {pred_part}  (conf={part_conf:.3f})")

    # ── Phase 2: fracture detection ───────────────────────────────
    try:
        pred_frac, frac_conf = run_fracture_inference(pred_part, image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Fracture inference failed")
        raise HTTPException(status_code=500, detail=f"Fracture model error: {e}")

    log.info(f"[{file.filename}] Fracture result: {pred_frac}  (sigmoid={frac_conf:.3f})")

    return PredictionResponse(
        filename             = file.filename or "unknown",
        predicted_part       = pred_part,
        part_confidence      = round(part_conf, 4),
        part_probabilities   = part_probs,
        predicted_fracture   = pred_frac,
        fracture_confidence  = round(frac_conf, 4),
        fracture_label       = "⚠ Fracture Detected" if pred_frac == "fractured" else "✓ No Fracture",
        is_fractured         = pred_frac == "fractured",
    )