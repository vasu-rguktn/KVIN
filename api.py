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

from  google import genai
from google.genai import types
import json
import re
from dotenv import load_dotenv

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
AI_THRESHOLD = 0.7  # confidence threshold for AI detection


load_dotenv()
# Gemini setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = None

if not GOOGLE_API_KEY:
    log.warning("⚠ GOOGLE_API_KEY not set — AI detection disabled")
else:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    log.info("✓ Gemini Client initialized")

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

    # AI detection
    is_ai_generated: bool
    ai_confidence: float
    ai_reason: Optional[str]


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
# AI Detection (Gemini)
# ─────────────────────────────────────────────────────────────────
def detect_ai_generated(image_bytes: bytes) -> dict:
    
    if client is None:
        return {
            "is_ai_generated": False,
            "confidence": 0.0,
            "reason": "Gemini not configured"
        }
    
    try:
        prompt = """
            You are an expert forensic AI system specializing in detecting AI-generated images, 
            with deep knowledge of image forensics, generative models, and visual artifacts. 
            Your role is to act as a highly skilled digital forensic analyst. 
            You must provide accurate and confident assessments of whether an image was AI-generated or authentic.

            Persona and Role:
            - You are a meticulous forensic investigator.
            - You cannot make assumptions without evidence from the image.
            - You always quantify your confidence as a float between 0.0 and 1.0.
            - You provide concise reasoning for your decision, highlighting visual cues, patterns, or artifacts.

            Process:
            1. Examine the image thoroughly for AI-specific artifacts such as:
            - unnatural textures
            - inconsistencies in lighting or anatomy
            - unusual backgrounds or details
            - generative model fingerprints (e.g., diffusion artifacts)
            2. Determine whether the image is AI-generated or authentic.
            3. Quantify your confidence in the detection (0 = completely unsure, 1 = completely certain).
            4. Provide a brief, clear explanation in the 'reason' field summarizing the key cues leading to your conclusion.
            5. Respond STRICTLY in JSON format ONLY. Do NOT include any commentary, greetings, or extra text outside the JSON.

            Required JSON format:
            {
                "is_ai_generated": true or false,        // boolean: true if AI-generated, false if authentic
                "confidence": 0.0 to 1.0,                // float: probability score of your assessment
                "reason": "short explanation"            // string: concise forensic reasoning
            }

            Additional Instructions:
            - JSON must be parseable by a standard JSON parser.
            - Use only boolean true/false, not strings like "True" or "False".
            - Keep 'reason' short, ideally 20-40 words max.
            - Focus on forensic evidence visible in the image.
        """

        for attempt in range(1):  # simple retry in case of transient errors
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[
                        prompt,
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg"
                        )
                    ]
                )
                text = response.text.strip()
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                    return {
                        "is_ai_generated": bool(result.get("is_ai_generated", False)),
                        "confidence": float(result.get("confidence", 0.0)),
                        "reason": result.get("reason", "").strip()
                    }
            except Exception as e:
                log.warning(f"Gemini attempt {attempt+1} failed: {e}")

    except Exception as e:
        log.exception("Gemini detection failed")
        return {
            "is_ai_generated": False,
            "confidence": 0.0,
            "reason": "AI detection unavailable",
        }


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
    if len(image_bytes) > 10_000_000:  # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large")
    
    # ── AI detection (NEW STEP) ─────────────────────────────
    ai_result = detect_ai_generated(image_bytes) or {
        "is_ai_generated": False,
        "confidence": 0.0,
        "reason": "AI detection unavailable"
    }

    log.info(
        f"[{file.filename}] AI detection → "
        f"is_ai={ai_result['is_ai_generated']} "
        f"conf={ai_result['confidence']:.3f}"
        f"reason={ai_result['reason']}"
    )

    # Strict blocking (only if high confidence)
    if ai_result["is_ai_generated"] and ai_result["confidence"] >= AI_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "AI-generated image detected",
                "confidence": ai_result["confidence"],
                "reason": ai_result["reason"],
            },
        )

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
        is_ai_generated      = ai_result["is_ai_generated"],
        ai_confidence        = round(ai_result["confidence"], 4),
        ai_reason            = ai_result.get("reason"),
    )