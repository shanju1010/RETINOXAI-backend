import io
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from .gradcam import generate_gradcam
from .model import CLASS_NAMES, load_checkpoint, prepare_pil
from .quality import assess_quality, enhance_fundus

from ..services.retinal_structure import (
    analyze_retinal_structures,
    annotate_retinal_structures,
    make_api_result,
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = Path(
    os.getenv(
        "RETINOXAI_MODEL",
        APP_ROOT / "models" / "best.pt",
    )
)

# Vercel Functions have a read-only deployed filesystem.
# Use /tmp for runtime-generated files. An environment variable allows
# the directory to be customized without changing the application code.
OUTPUT_DIR = Path(
    os.getenv(
        "RETINOXAI_OUTPUT_DIR",
        "/tmp/retinoxai_outputs",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCREENING_RESULT_FILE = OUTPUT_DIR / "screening_result.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RETINOXAI Prototype API",
    version="0.4.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL STATE
# ============================================================

_model = None

_latest_cam = None
_latest_structure = None
_latest_enhanced = None


# ============================================================
# MODEL LOADING
# ============================================================

def get_model():
    global _model

    if _model is None:
        _model = load_checkpoint(
            str(CHECKPOINT),
            DEVICE,
        )

    return _model


# ============================================================
# JSON HELPERS
# ============================================================

def _json_safe(value):
    """
    Convert NumPy / Torch scalar values into normal Python
    values so the result can always be written to JSON.
    """
    if isinstance(value, dict):
        return {
            str(key): _json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()

    return value


def save_screening_result(result: dict):
    """
    Save the latest real screening result for MATLAB/Simulink
    integration and local inspection.

    File:
        backend/outputs/screening_result.json
    """
    safe_result = _json_safe(result)

    temp_file = SCREENING_RESULT_FILE.with_suffix(".tmp")

    try:
        with open(
            temp_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                safe_result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        temp_file.replace(
            SCREENING_RESULT_FILE
        )

    except Exception as exc:
        print(
            f"Warning: could not save screening result: {exc}"
        )

        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "EfficientNet-B0",
        "checkpoint_exists": CHECKPOINT.exists(),
        "retinal_structure_analysis": True,
        "enhanced_image_endpoint": True,
        "gradcam_endpoint": True,
        "screening_result_export": True,
    }


# ============================================================
# LATEST SCREENING RESULT
# ============================================================

@app.get("/screening-result/latest")
def latest_screening_result():
    """
    Return the latest screening_result.json through the API.
    Useful for testing the FastAPI -> MATLAB integration.
    """

    if not SCREENING_RESULT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No screening result exists yet. "
                "Run /predict first."
            ),
        )

    try:
        with open(
            SCREENING_RESULT_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            result = json.load(f)

        return JSONResponse(content=result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not read screening result: {exc}"
            ),
        ) from exc


# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    global _latest_cam
    global _latest_structure
    global _latest_enhanced

    # --------------------------------------------------------
    # 1. Validate uploaded file
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        # ----------------------------------------------------
        # 2. Read uploaded image
        # ----------------------------------------------------

        raw = await file.read()

        if not raw:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        try:
            original = Image.open(
                io.BytesIO(raw)
            ).convert("RGB")

        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            ) from exc

        # PIL RGB -> NumPy BGR
        bgr = np.array(
            original
        )[:, :, ::-1].copy()

        # ----------------------------------------------------
        # 3. Image Quality Assessment
        # ----------------------------------------------------

        quality = assess_quality(bgr)

        # ----------------------------------------------------
        # 3A. Retake flow
        # ----------------------------------------------------

        if not quality["acceptable"]:
            # Clear previous successful-session images so the
            # frontend cannot accidentally reuse stale results.
            _latest_cam = None
            _latest_structure = None
            _latest_enhanced = None

            retake_result = {
                "timestamp": datetime.now().isoformat(),
                "status": "retake",
                "message": (
                    "Image quality is below the prototype "
                    "threshold. Please capture or upload a "
                    "clearer fundus image."
                ),
                "filename": file.filename,
                "quality": quality,
                "retinal_structure": None,
                "explanation_image": None,
                "retinal_structure_image": None,
                "enhanced_image": None,
                "note": (
                    "AI grading was not performed because "
                    "the image did not pass the prototype "
                    "quality gate."
                ),
            }

            # Save retake result too, so MATLAB/Simulink can
            # see that the latest screening stopped at quality.
            save_screening_result(
                retake_result
            )

            return retake_result

        # ----------------------------------------------------
        # 4. Retinal Structure Analysis
        # ----------------------------------------------------
        #
        # Outputs:
        #   - vessel analysis
        #   - optic disc localization
        #   - IDRiD-calibrated fovea estimation
        # ----------------------------------------------------

        structure_result = analyze_retinal_structures(
            bgr
        )

        # Create annotated retinal image
        annotated_structure = annotate_retinal_structures(
            bgr,
            structure_result,
        )

        # Encode annotated structure image
        success, structure_encoded = cv2.imencode(
            ".png",
            annotated_structure,
        )

        if not success:
            raise RuntimeError(
                "Could not encode retinal structure image."
            )

        _latest_structure = (
            structure_encoded.tobytes()
        )

        # Convert structure result to JSON-safe values
        structure_api = make_api_result(
            structure_result
        )

        # ----------------------------------------------------
        # 5. Adaptive Enhancement
        # ----------------------------------------------------

        enhanced_bgr = enhance_fundus(
            bgr
        )

        # ----------------------------------------------------
        # 5A. Store enhanced image for frontend workflow
        # ----------------------------------------------------

        success, enhanced_encoded = cv2.imencode(
            ".png",
            enhanced_bgr,
        )

        if not success:
            raise RuntimeError(
                "Could not encode enhanced image."
            )

        _latest_enhanced = (
            enhanced_encoded.tobytes()
        )

        # BGR -> RGB for PIL
        enhanced_rgb = (
            enhanced_bgr[:, :, ::-1]
        )

        enhanced = Image.fromarray(
            enhanced_rgb
        )

        # ----------------------------------------------------
        # 6. EfficientNet-B0 DR Inference
        # ----------------------------------------------------

        model = get_model()

        tensor = prepare_pil(
            enhanced
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)

            probs = torch.softmax(
                logits,
                dim=1,
            )[0].cpu().numpy()

        # Predicted class
        pred = int(
            np.argmax(probs)
        )

        confidence = float(
            probs[pred]
        )

        # Grade 2+ = referable DR
        referable = bool(pred >= 2)

        # ----------------------------------------------------
        # 7. Grad-CAM
        # ----------------------------------------------------

        _, cam_img = generate_gradcam(
            model,
            tensor,
            enhanced,
        )

        cam_buffer = io.BytesIO()

        cam_img.save(
            cam_buffer,
            format="PNG",
        )

        _latest_cam = (
            cam_buffer.getvalue()
        )

        # ----------------------------------------------------
        # 8. Build real screening result
        # ----------------------------------------------------

        screening_result = {
            "timestamp": datetime.now().isoformat(),

            "status": "ok",

            "filename": file.filename,

            "prediction": {
                "grade": pred,
                "grade_name": CLASS_NAMES[pred],
                "confidence": round(
                    confidence,
                    4,
                ),
                "referable_dr": referable,
            },

            "quality": quality,

            "retinal_structure": structure_api,

            "device": str(DEVICE),

            "model": "EfficientNet-B0",

            "outputs": {
                "gradcam": "/explanation/latest",
                "enhanced_image": "/enhanced/latest",
                "retinal_structure": (
                    "/retinal-structure/latest"
                ),
            },

            "workflow": {
                "image_acquisition": True,
                "quality_assessment": True,
                "adaptive_enhancement": True,
                "ai_inference": True,
                "gradcam": True,
                "retinal_structure_analysis": True,
            },

            "note": (
                "Prototype screening support only; "
                "not a clinical diagnosis."
            ),
        }

        # ----------------------------------------------------
        # 9. SAVE RESULT FOR MATLAB / SIMULINK
        # ----------------------------------------------------

        save_screening_result(
            screening_result
        )

        # ----------------------------------------------------
        # 10. RETURN API RESPONSE
        # ----------------------------------------------------

        return screening_result

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc


# ============================================================
# GRAD-CAM IMAGE
# ============================================================

@app.get("/explanation/latest")
def latest_explanation():

    if _latest_cam is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No explanation generated yet. "
                "Run /predict first."
            ),
        )

    return StreamingResponse(
        io.BytesIO(
            _latest_cam
        ),
        media_type="image/png",
        headers={
            "Cache-Control": (
                "no-cache, no-store, must-revalidate"
            ),
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ============================================================
# RETINAL STRUCTURE ANNOTATION
# ============================================================

@app.get("/retinal-structure/latest")
def latest_retinal_structure():

    if _latest_structure is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No retinal structure analysis "
                "generated yet. Run /predict first."
            ),
        )

    return StreamingResponse(
        io.BytesIO(
            _latest_structure
        ),
        media_type="image/png",
        headers={
            "Cache-Control": (
                "no-cache, no-store, must-revalidate"
            ),
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ============================================================
# ENHANCED FUNDUS IMAGE
# ============================================================

@app.get("/enhanced/latest")
def latest_enhanced():

    if _latest_enhanced is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No enhanced image generated yet. "
                "Run /predict first."
            ),
        )

    return StreamingResponse(
        io.BytesIO(
            _latest_enhanced
        ),
        media_type="image/png",
        headers={
            "Cache-Control": (
                "no-cache, no-store, must-revalidate"
            ),
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
