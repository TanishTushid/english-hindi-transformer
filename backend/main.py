"""
FastAPI backend — English → Hindi Translator
Two models:
  nllb    : Your fine-tuned NLLB model (models/nllb_final/)
  scratch : Your trained from-scratch Transformer (models/transformer_trained.weights.h5)
"""

import os, sys, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ─── Silence TF output ────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR          = Path(__file__).parent
ROOT                 = BACKEND_DIR.parent
MODELS_DIR           = ROOT / "models"
FRONTEND_DIR         = ROOT / "frontend"

NLLB_MODEL_PATH      = str(MODELS_DIR / "nllb_final")
SCRATCH_WEIGHTS_PATH = str(MODELS_DIR / "transformer_trained.weights.h5")
SCRATCH_CONFIG_PATH  = str(MODELS_DIR / "transformer_config.json")
ENG_SP_MODEL         = str(MODELS_DIR / "english_sp.model")
HIN_SP_MODEL         = str(MODELS_DIR / "hindi_sp.model")

# Add backend dir to sys.path so model.py is importable
sys.path.insert(0, str(BACKEND_DIR))

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="English → Hindi Translator API",
    description=(
        "Translate English text to Hindi using:\n"
        "- **nllb**: Your fine-tuned NLLB-200 model\n"
        "- **scratch**: Your custom Transformer trained from scratch"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_nllb_translator    = None
_scratch_translator = None
_model_availability = {}


def get_nllb():
    global _nllb_translator
    if _nllb_translator is None:
        from translate_nllb import NLLBTranslator
        _nllb_translator = NLLBTranslator(NLLB_MODEL_PATH)
    return _nllb_translator


def get_scratch():
    global _scratch_translator
    if _scratch_translator is None:
        from translate_scratch import ScratchTranslator
        _scratch_translator = ScratchTranslator(
            SCRATCH_WEIGHTS_PATH, SCRATCH_CONFIG_PATH,
            ENG_SP_MODEL, HIN_SP_MODEL,
        )
    return _scratch_translator


# ─── Startup check ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global _model_availability

    try:
        import torch
        torch_ok = True
    except ImportError:
        torch_ok = False

    try:
        import tensorflow
        tf_ok = True
    except ImportError:
        tf_ok = False

    try:
        import sentencepiece
        sp_ok = True
    except ImportError:
        sp_ok = False

    nllb_available = (
        torch_ok and
        os.path.exists(NLLB_MODEL_PATH) and
        os.path.exists(os.path.join(NLLB_MODEL_PATH, "model.safetensors"))
    )

    scratch_available = (
        tf_ok and sp_ok and
        os.path.exists(SCRATCH_WEIGHTS_PATH) and
        os.path.exists(SCRATCH_CONFIG_PATH) and
        os.path.exists(ENG_SP_MODEL) and
        os.path.exists(HIN_SP_MODEL)
    )

    def reason(name):
        if name == "nllb":
            if not torch_ok:  return "PyTorch not installed"
            if not os.path.exists(NLLB_MODEL_PATH): return f"Path missing: {NLLB_MODEL_PATH}"
            if not os.path.exists(os.path.join(NLLB_MODEL_PATH,"model.safetensors")): return "model.safetensors missing"
            return "ready"
        else:
            if not tf_ok:     return "TensorFlow not installed"
            if not sp_ok:     return "sentencepiece not installed"
            if not os.path.exists(SCRATCH_WEIGHTS_PATH): return "transformer_trained.weights.h5 missing"
            if not os.path.exists(SCRATCH_CONFIG_PATH):  return "transformer_config.json missing"
            if not os.path.exists(ENG_SP_MODEL): return f"english_sp.model missing — run generate_tokenizers.py"
            if not os.path.exists(HIN_SP_MODEL): return f"hindi_sp.model missing — run generate_tokenizers.py"
            return "ready"

    _model_availability = {
        "nllb": {
            "available": nllb_available,
            "path": NLLB_MODEL_PATH,
            "status": reason("nllb"),
        },
        "scratch": {
            "available": scratch_available,
            "weights": SCRATCH_WEIGHTS_PATH,
            "config": SCRATCH_CONFIG_PATH,
            "eng_tokenizer": ENG_SP_MODEL,
            "hin_tokenizer": HIN_SP_MODEL,
            "status": reason("scratch"),
        },
    }

    print("\n=== Model Availability ===")
    for name, info in _model_availability.items():
        state = "available" if info["available"] else "unavailable"
        print(f"  [{state}] {name:12s} - {info['status']}")
    print("===========================\n")


# ─── Schemas ──────────────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str
    model: str = "nllb"
    max_tokens: int = Field(default=256, gt=0, le=1024)


class TranslateResponse(BaseModel):
    translation: str
    model_used: str
    time_ms: float
    input_text: str


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "models": _model_availability}


@app.get("/api/models")
def get_models():
    return _model_availability


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Input text cannot be empty.")
    if len(text) > 1000:
        raise HTTPException(400, "Input text too long (max 1000 characters).")

    model_key = req.model.lower()
    if model_key not in ("nllb", "scratch"):
        raise HTTPException(400, "model must be 'nllb' or 'scratch'.")

    info = _model_availability.get(model_key, {})
    if not info.get("available", False):
        status = info.get("status", "unknown reason")
        raise HTTPException(
            503,
            f"Model '{model_key}' is not available: {status}"
        )

    t0 = time.perf_counter()
    try:
        if model_key == "nllb":
            translation = get_nllb().translate(text, max_new_tokens=req.max_tokens)
        else:
            translation = get_scratch().translate(text)
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {str(e)}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return TranslateResponse(
        translation=translation,
        model_used=model_key,
        time_ms=round(elapsed_ms, 2),
        input_text=text,
    )


# ─── Serve frontend ───────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
