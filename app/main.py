"""
app/main.py – Servicio de inferencia con FastAPI.

Endpoints:
  GET  /          → info de la API
  GET  /health    → health check (usado por K8s readiness probe)
  POST /predict   → predicción de especie Iris
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import IrisModel

# ─── Lifespan ────────────────────────────────────────────────────────────────
_model: IrisModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = IrisModel()
    print("✅ Modelo cargado y listo.")
    yield
    _model = None
    print("🛑 Servicio apagado.")


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Servicio de inferencia para clasificar especies de iris.\n\n"
        "Taller MLOps – UNI 2026"
    ),
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan,
)

START_TIME = time.time()


# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Largo del sépalo (cm)")
    sepal_width:  float = Field(..., gt=0, description="Ancho del sépalo (cm)")
    petal_length: float = Field(..., gt=0, description="Largo del pétalo (cm)")
    petal_width:  float = Field(..., gt=0, description="Ancho del pétalo (cm)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width":  3.5,
                "petal_length": 1.4,
                "petal_width":  0.2,
            }
        }
    }


class PredictResponse(BaseModel):
    class_id:      int
    class_name:    str
    confidence:    float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status:   str
    uptime_s: float
    version:  str


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Iris Classifier API – ver /docs para la documentación interactiva",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    """Kubernetes readiness/liveness probe."""
    return HealthResponse(
        status="ok",
        uptime_s=round(time.time() - START_TIME, 2),
        version=app.version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    """
    Clasifica una flor de iris a partir de sus medidas en centímetros.

    Retorna la especie predicha y la distribución de probabilidades.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    features = [[
        req.sepal_length,
        req.sepal_width,
        req.petal_length,
        req.petal_width,
    ]]
    result = _model.predict(features)
    return PredictResponse(**result)
