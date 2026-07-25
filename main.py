"""
Aldora AI Models Microservice
FastAPI — CheXpert, ECG-FM, HAM10000, XGBoost Sepse
"""
import os, io, base64, logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aldora-ai")

# ── Lazy model registry ─────────────────────────────────────────────────────
MODEL_REGISTRY: dict = {}

def load_models():
    """Carrega modelos sob demanda para não bloquear o startup."""
    try:
        from models.chexpert import load_chexpert
        MODEL_REGISTRY["chexpert"] = load_chexpert()
        logger.info("CheXpert carregado.")
    except Exception as e:
        logger.warning(f"CheXpert não disponível: {e}")

    try:
        from models.ecg import load_ecg
        MODEL_REGISTRY["ecg"] = load_ecg()
        logger.info("ECG-FM carregado.")
    except Exception as e:
        logger.warning(f"ECG-FM não disponível: {e}")

    try:
        from models.derma import load_derma
        MODEL_REGISTRY["derma"] = load_derma()
        logger.info("HAM10000/EfficientNet carregado.")
    except Exception as e:
        logger.warning(f"Derma não disponível: {e}")

    try:
        from models.sepse import load_sepse
        MODEL_REGISTRY["sepse"] = load_sepse()
        logger.info("XGBoost Sepse carregado.")
    except Exception as e:
        logger.warning(f"Sepse não disponível: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="Aldora AI Models",
    version="1.0.0",
    description="Microserviço de modelos de IA médica: CheXpert, ECG-FM, HAM10000, XGBoost Sepse",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.aldora.com.br", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Modelos de entrada ───────────────────────────────────────────────────────

class ImageRequest(BaseModel):
    image_base64: str          # base64 da imagem (PNG/JPEG)
    image_mime: str = "image/jpeg"
    paciente_id: Optional[str] = None
    metadados: Optional[dict] = None


class ECGRequest(BaseModel):
    sinal: list[float]         # sinal ECG normalizado (5000 amostras, 500Hz, 12 derivações flatten)
    frequencia_hz: int = 500
    derivacoes: int = 12
    paciente_id: Optional[str] = None
    metadados: Optional[dict] = None


class SepseRequest(BaseModel):
    # 40 variáveis do PhysioNet Challenge 2019
    hr: Optional[float] = None          # Heart Rate (bpm)
    o2sat: Optional[float] = None       # SpO2 (%)
    temp: Optional[float] = None        # Temperatura (°C)
    sbp: Optional[float] = None         # Pressão sistólica
    map: Optional[float] = None         # Pressão arterial média
    dbp: Optional[float] = None         # Pressão diastólica
    resp: Optional[float] = None        # Frequência respiratória
    etco2: Optional[float] = None
    baseexcess: Optional[float] = None
    hco3: Optional[float] = None
    ph: Optional[float] = None
    paco2: Optional[float] = None
    sao2: Optional[float] = None
    ast: Optional[float] = None
    bun: Optional[float] = None
    alkalinephos: Optional[float] = None
    calcium: Optional[float] = None
    chloride: Optional[float] = None
    creatinine: Optional[float] = None
    bilirubin_direct: Optional[float] = None
    glucose: Optional[float] = None
    lactate: Optional[float] = None
    magnesium: Optional[float] = None
    phosphate: Optional[float] = None
    potassium: Optional[float] = None
    bilirubin_total: Optional[float] = None
    trop_i: Optional[float] = None
    hct: Optional[float] = None
    hgb: Optional[float] = None
    ptt: Optional[float] = None
    wbc: Optional[float] = None
    fibrinogen: Optional[float] = None
    platelets: Optional[float] = None
    age: Optional[float] = None
    gender: Optional[float] = None       # 0=F, 1=M
    unit1: Optional[float] = None        # Unidade MICU=1
    unit2: Optional[float] = None        # Unidade SICU=1
    hosp_adm_elapsed: Optional[float] = None
    icu_los_days: Optional[float] = None
    hora_no_icu: Optional[float] = None
    paciente_id: Optional[str] = None


DISCLAIMER_CFM = (
    "AVISO REGULATÓRIO — CFM Resolução 2.454/2026: Este resultado é gerado por sistema de apoio diagnóstico "
    "(SaMD/ANVISA). NÃO substitui avaliação clínica por profissional habilitado. "
    "Não comunicar ao paciente sem mediação do médico responsável."
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos": {k: "carregado" for k in MODEL_REGISTRY},
        "versao": "1.0.0",
    }


@app.post("/v1/chexpert")
async def chexpert(req: ImageRequest):
    """
    CheXpert/ChestX-ray14 — DenseNet121, 14 patologias pulmonares.
    AUC médio ~0.84. Treinado em CheXpert (224,316 imagens).
    """
    modelo = MODEL_REGISTRY.get("chexpert")
    if modelo is None:
        raise HTTPException(503, "Modelo CheXpert não carregado. Verifique requirements e HUGGINGFACE_TOKEN.")

    try:
        img_bytes = base64.b64decode(req.image_base64)
        resultado = modelo.predict(img_bytes)
    except Exception as e:
        logger.exception("Erro CheXpert")
        raise HTTPException(500, f"Erro na inferência: {e}")

    return {
        "patologias": resultado,
        "modelo": "torchxrayvision/densenet121-res224-chex",
        "disclaimer": DISCLAIMER_CFM,
    }


@app.post("/v1/ecg")
async def ecg(req: ECGRequest):
    """
    ECG-FM (bowang-lab/ecg-fm) — modelo fundação para ECG.
    Detecta arritmias, isquemia e achados estruturais.
    """
    modelo = MODEL_REGISTRY.get("ecg")
    if modelo is None:
        raise HTTPException(503, "Modelo ECG-FM não carregado.")

    try:
        resultado = modelo.predict(req.sinal, req.derivacoes, req.frequencia_hz)
    except Exception as e:
        logger.exception("Erro ECG-FM")
        raise HTTPException(500, f"Erro na inferência: {e}")

    return {
        "achados": resultado,
        "modelo": "bowang-lab/ecg-fm",
        "disclaimer": DISCLAIMER_CFM,
    }


@app.post("/v1/derma")
async def derma(req: ImageRequest):
    """
    HAM10000 — EfficientNet-B4, 7 classes de lesões cutâneas.
    Acurácia ~90%. Classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC.
    """
    modelo = MODEL_REGISTRY.get("derma")
    if modelo is None:
        raise HTTPException(503, "Modelo Dermatologia não carregado.")

    try:
        img_bytes = base64.b64decode(req.image_base64)
        resultado = modelo.predict(img_bytes)
    except Exception as e:
        logger.exception("Erro Derma")
        raise HTTPException(500, f"Erro na inferência: {e}")

    return {
        "classes": resultado,
        "modelo": "efficientnet-b4-ham10000",
        "disclaimer": DISCLAIMER_CFM,
    }


@app.post("/v1/sepse")
async def sepse(req: SepseRequest):
    """
    XGBoost Sepse — PhysioNet Challenge 2019 (60k pacientes ICU).
    AUC estimado 0.82-0.88. 40 variáveis laboratoriais e vitais.
    """
    modelo = MODEL_REGISTRY.get("sepse")
    if modelo is None:
        raise HTTPException(503, "Modelo Sepse não carregado.")

    try:
        features = _extrair_features_sepse(req)
        resultado = modelo.predict(features)
    except Exception as e:
        logger.exception("Erro Sepse")
        raise HTTPException(500, f"Erro na inferência: {e}")

    return {
        "probabilidade_sepse": resultado["prob"],
        "risco": resultado["risco"],    # baixo / moderado / alto / critico
        "score": resultado["score"],
        "variaveis_impacto": resultado.get("shap_top5", []),
        "modelo": "xgboost-sepse-physionet2019",
        "disclaimer": DISCLAIMER_CFM,
    }


def _extrair_features_sepse(req: SepseRequest) -> list[float]:
    campos = [
        "hr", "o2sat", "temp", "sbp", "map", "dbp", "resp", "etco2",
        "baseexcess", "hco3", "ph", "paco2", "sao2", "ast", "bun",
        "alkalinephos", "calcium", "chloride", "creatinine", "bilirubin_direct",
        "glucose", "lactate", "magnesium", "phosphate", "potassium",
        "bilirubin_total", "trop_i", "hct", "hgb", "ptt", "wbc",
        "fibrinogen", "platelets", "age", "gender", "unit1", "unit2",
        "hosp_adm_elapsed", "icu_los_days", "hora_no_icu",
    ]
    return [float(getattr(req, c) or 0.0) for c in campos]
