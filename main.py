"""
Aldora AI Models Microservice v2
FastAPI — CheXpert, ECG (rule-based), HAM10000/EfficientNet-B4, XGBoost Sepse, Pill Identifier
Todos os endpoints retornam: {resultado, confianca, modelo, aviso, timestamp}
"""
import os, io, base64, logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aldora-ai")

MODEL_REGISTRY: dict = {}

DISCLAIMER_CFM = (
    "AVISO REGULATÓRIO — CFM Resolução 2.454/2026: Este resultado é gerado por sistema de apoio diagnóstico "
    "(SaMD/ANVISA). NÃO substitui avaliação clínica por profissional habilitado. "
    "Não comunicar ao paciente sem mediação do médico responsável."
)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_models():
    """Carrega todos os modelos ao iniciar. Falhas individuais não impedem o startup."""
    try:
        from models.chexpert_model import load_chexpert
        MODEL_REGISTRY["chexpert"] = load_chexpert()
        logger.info("CheXpert (densenet121-res224-all) carregado.")
    except Exception as e:
        logger.warning(f"CheXpert não disponível: {e}")

    try:
        from models.ecg_model import load_ecg
        MODEL_REGISTRY["ecg"] = load_ecg()
        logger.info("ECG (rule-based clínico) carregado.")
    except Exception as e:
        logger.warning(f"ECG não disponível: {e}")

    try:
        from models.derma_model import load_derma
        MODEL_REGISTRY["derma"] = load_derma()
        logger.info("HAM10000/EfficientNet-B4 carregado.")
    except Exception as e:
        logger.warning(f"Derma não disponível: {e}")

    try:
        from models.sepse_model import load_sepse
        MODEL_REGISTRY["sepse"] = load_sepse()
        logger.info("XGBoost Sepse carregado.")
    except Exception as e:
        logger.warning(f"Sepse não disponível: {e}")

    try:
        from models.pill_model import load_pill
        MODEL_REGISTRY["pill"] = load_pill()
        logger.info("Pill Identifier (Claude Vision) carregado.")
    except Exception as e:
        logger.warning(f"Pill Identifier não disponível: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="Aldora AI Models",
    version="2.0.0",
    description=(
        "Microserviço de IA médica: CheXpert (14 patologias), "
        "ECG rule-based, HAM10000 (7 classes), XGBoost Sepse, Pill Identifier. "
        "CFM 2.454/2026 — apoio diagnóstico, não substituição clínica."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.aldora.com.br", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas de entrada ────────────────────────────────────────────────────────

class ImageRequest(BaseModel):
    image_base64: str
    image_mime: str = "image/jpeg"
    paciente_id: Optional[str] = None
    metadados: Optional[dict] = None


class ECGRequest(BaseModel):
    fc: float               # Frequência cardíaca (bpm)
    pr_ms: float            # Intervalo PR (ms)
    qrs_ms: float           # Duração QRS (ms)
    qtc_ms: float           # QTc corrigido por Bazett (ms)
    rr_variability: float = 0.0  # Variabilidade RR (ms); 0 se indisponível
    paciente_id: Optional[str] = None


class SepseRequest(BaseModel):
    hr: Optional[float] = None
    o2sat: Optional[float] = None
    temp: Optional[float] = None
    sbp: Optional[float] = None
    map: Optional[float] = None
    dbp: Optional[float] = None
    resp: Optional[float] = None
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
    gender: Optional[float] = None      # 0=F, 1=M
    unit1: Optional[float] = None       # MICU=1
    unit2: Optional[float] = None       # SICU=1
    hosp_adm_elapsed: Optional[float] = None
    icu_los_days: Optional[float] = None
    hora_no_icu: Optional[float] = None
    paciente_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos": {k: "carregado" for k in MODEL_REGISTRY},
        "versao": "2.0.0",
        "timestamp": _ts(),
    }


@app.post("/v1/chexpert")
async def chexpert(req: ImageRequest):
    """
    CheXpert — torchxrayvision DenseNet121-res224-all.
    14 patologias pulmonares. AUC médio ~0.84.
    """
    modelo = MODEL_REGISTRY.get("chexpert")
    if modelo is None:
        raise HTTPException(503, "Modelo CheXpert não carregado. Verifique os logs de startup.")

    try:
        img_bytes = base64.b64decode(req.image_base64)
        resultado = modelo.predict(img_bytes)
    except Exception as e:
        logger.exception("Erro CheXpert")
        raise HTTPException(500, f"Erro na inferência: {e}")

    top_conf = resultado[0]["probabilidade"] if resultado else 0.0
    return {
        "resultado": resultado,
        "confianca": round(top_conf, 4),
        "modelo": "torchxrayvision/densenet121-res224-all",
        "aviso": DISCLAIMER_CFM,
        "timestamp": _ts(),
    }


@app.post("/v1/ecg")
async def ecg(req: ECGRequest):
    """
    ECG — análise rule-based clínica (ACC/AHA/SBC).
    Parâmetros: FC, intervalo PR, duração QRS, QTc (Bazett), variabilidade RR.
    """
    modelo = MODEL_REGISTRY.get("ecg")
    if modelo is None:
        raise HTTPException(503, "Modelo ECG não carregado.")

    try:
        resultado = modelo.predict(req.fc, req.pr_ms, req.qrs_ms, req.qtc_ms, req.rr_variability)
    except Exception as e:
        logger.exception("Erro ECG")
        raise HTTPException(500, f"Erro na análise ECG: {e}")

    achados = resultado.get("achados", [])
    top_conf = achados[0]["probabilidade"] if achados else 0.0
    return {
        "resultado": resultado,
        "confianca": round(top_conf, 4),
        "modelo": "rule-based-ecg-v1",
        "aviso": DISCLAIMER_CFM,
        "timestamp": _ts(),
    }


@app.post("/v1/derma")
async def derma(req: ImageRequest):
    """
    HAM10000 — EfficientNet-B4 (timm).
    7 classes de lesões cutâneas. Acurácia ~90% no HAM10000 test set.
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

    top_conf = resultado[0]["probabilidade"] if resultado else 0.0
    return {
        "resultado": resultado,
        "confianca": round(top_conf, 4),
        "modelo": "efficientnet_b4-ham10000",
        "aviso": DISCLAIMER_CFM,
        "timestamp": _ts(),
    }


@app.post("/v1/sepse")
async def sepse(req: SepseRequest):
    """
    XGBoost Sepse — PhysioNet Challenge 2019 (≈60k pacientes UTI).
    40 variáveis laboratoriais e sinais vitais. AUC estimado 0.82–0.88.
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
        "resultado": resultado,
        "confianca": resultado["prob"],
        "modelo": "xgboost-sepse-physionet2019",
        "aviso": DISCLAIMER_CFM,
        "timestamp": _ts(),
    }


@app.post("/v1/pill")
async def pill(req: ImageRequest):
    """
    Pill Identifier — Claude claude-sonnet-4-6 Vision.
    Identifica comprimidos, cápsulas e medicamentos por imagem.
    Requer ANTHROPIC_API_KEY configurada no ambiente.
    """
    modelo = MODEL_REGISTRY.get("pill")
    if modelo is None:
        raise HTTPException(503, "Pill Identifier não carregado. Verifique ANTHROPIC_API_KEY.")

    try:
        img_bytes = base64.b64decode(req.image_base64)
        resultado = modelo.predict(img_bytes, req.image_mime)
    except Exception as e:
        logger.exception("Erro Pill Identifier")
        raise HTTPException(500, f"Erro na identificação: {e}")

    confianca_num = {"alta": 0.90, "media": 0.65, "baixa": 0.35}
    nivel = resultado.get("confianca", "baixa")
    return {
        "resultado": resultado,
        "confianca": confianca_num.get(nivel, 0.35),
        "modelo": "claude-sonnet-4-6-vision",
        "aviso": DISCLAIMER_CFM,
        "timestamp": _ts(),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extrair_features_sepse(req: SepseRequest) -> list:
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
