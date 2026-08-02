"""
Aldora AI Models Microservice v3
FastAPI — 5 modelos legados + 11 tabular GBM/XGBoost + 7 ONNX imagem
CFM 2.454/2026 — apoio diagnóstico, não substituição clínica.
"""
import io
import os
import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aldora-ai")

MODEL_REGISTRY: dict = {}
_PTH_MODEL_CACHE: dict = {}  # lazy-cache para modelos PyTorch carregados sob demanda

DISCLAIMER_CFM = (
    "AVISO REGULATÓRIO — CFM Resolução 2.454/2026: Este resultado é gerado por sistema de apoio diagnóstico "
    "(SaMD/ANVISA). NÃO substitui avaliação clínica por profissional habilitado. "
    "Não comunicar ao paciente sem mediação do médico responsável."
)

DISCLAIMER_SHORT = (
    "Ferramenta de apoio à decisão clínica. Decisão final é do médico. CFM 2.454/2026."
)

# Labels para modelos ONNX (ordem dos neurônios de saída)
_ONNX_LABELS: dict[str, list[str]] = {
    "chest_xray":  ["normal", "anormal"],
    "skin":        ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"],
    "retinopathy": ["grau_0", "grau_1", "grau_2", "grau_3", "grau_4"],
    "brain_tumor": ["glioma", "meningioma", "pituitary", "no_tumor"],
    "fracture":    ["normal", "fratura"],
    "glaucoma":    ["normal", "glaucoma"],
    "mammography": ["normal", "anormal"],
}


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Helpers de inferência ─────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _preprocess_onnx(image_bytes: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
        / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,224,224)


def _onnx_infer(session, image_bytes: bytes, labels: list[str], model_file: str) -> dict:
    inp = _preprocess_onnx(image_bytes)

    if isinstance(session, str):
        # PTH path — inferência PyTorch com lazy-cache
        import torch
        if session not in _PTH_MODEL_CACHE:
            model = torch.load(session, map_location="cpu", weights_only=False)
            model.eval()
            _PTH_MODEL_CACHE[session] = model
        pth_model = _PTH_MODEL_CACHE[session]
        t = torch.tensor(inp)
        with torch.no_grad():
            logits = pth_model(t)
            out = torch.softmax(logits, dim=1)[0].cpu().numpy()
    else:
        # ONNX InferenceSession
        out = session.run(None, {session.get_inputs()[0].name: inp})[0][0]
        if abs(float(out.sum()) - 1.0) > 0.05:
            out = _softmax(out)

    pred_idx = int(out.argmax())
    return {
        "predicao":      labels[pred_idx],
        "confianca":     round(float(out[pred_idx]), 4),
        "scores":        {labels[i]: round(float(out[i]), 4) for i in range(len(labels))},
        "modelo_versao": model_file,
        "disclaimer":    DISCLAIMER_SHORT,
    }


def _tabular_infer(model, values: list, features: list[str], model_file: str) -> dict:
    # BUG 2 FIX: InferenceSession não tem .predict() — usa .run()
    try:
        from onnxruntime import InferenceSession as _OrtSession
        if isinstance(model, _OrtSession):
            input_name = model.get_inputs()[0].name
            arr = np.array([values], dtype=np.float32)
            outputs = model.run(None, {input_name: arr})
            # XGBoost ONNX: outputs[0]=labels, outputs[1]=probs (dict ou array)
            if len(outputs) >= 2:
                proba_raw = outputs[1]
                if isinstance(proba_raw, list) and proba_raw and isinstance(proba_raw[0], dict):
                    p1 = float(proba_raw[0].get(1, proba_raw[0].get("1", 0.5)))
                    p0 = 1.0 - p1
                elif hasattr(proba_raw, "shape") and len(proba_raw.shape) == 2:
                    p0, p1 = float(proba_raw[0][0]), float(proba_raw[0][1])
                else:
                    p1 = float(outputs[0][0])
                    p0 = 1.0 - p1
            else:
                p1 = float(outputs[0][0])
                p0 = 1.0 - p1
            label = "1" if p1 >= 0.5 else "0"
            return {
                "predicao":      label,
                "confianca":     round(max(p0, p1), 4),
                "scores":        {"0": round(p0, 4), "1": round(p1, 4)},
                "modelo_versao": model_file,
                "disclaimer":    DISCLAIMER_SHORT,
            }
    except ImportError:
        pass

    import pandas as pd
    X = pd.DataFrame([dict(zip(features, values))])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = [str(c) for c in (model.classes_ if hasattr(model, "classes_") else range(len(proba)))]
        idx = int(proba.argmax())
        return {
            "predicao":      classes[idx],
            "confianca":     round(float(proba[idx]), 4),
            "scores":        {c: round(float(p), 4) for c, p in zip(classes, proba)},
            "modelo_versao": model_file,
            "disclaimer":    DISCLAIMER_SHORT,
        }
    pred = float(model.predict(X)[0])
    label = "1" if pred >= 0.5 else "0"
    return {
        "predicao":      label,
        "confianca":     round(abs(pred - 0.5) + 0.5, 4),
        "scores":        {"0": round(1.0 - pred, 4), "1": round(pred, 4)},
        "modelo_versao": model_file,
        "disclaimer":    DISCLAIMER_SHORT,
    }


# ── Carregamento de modelos ───────────────────────────────────────────────────

def _load_pth(key: str, path: str, label: str) -> None:
    """Carrega modelo PyTorch (.pth) via torch.load — apenas verifica se arquivo existe e é válido."""
    try:
        import torch
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            raise FileNotFoundError(f"arquivo ausente ou vazio: {path}")
        # Registra o path — inferência real carrega no endpoint
        MODEL_REGISTRY[key] = path
        logger.info("%s carregado (%s).", label, path)
    except Exception as e:
        logger.warning("%s não disponível: %s", label, e)

def _load_pkl(key: str, path: str, label: str) -> None:
    try:
        import joblib
        MODEL_REGISTRY[key] = joblib.load(path)
        logger.info("%s carregado (%s).", label, path)
    except Exception as e:
        logger.warning("%s não disponível: %s", label, e)


def _load_onnx_gbm(key: str, path: str, label: str) -> None:
    """Carrega modelo GBM exportado como ONNX via onnxruntime."""
    try:
        from onnxruntime import InferenceSession
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            raise FileNotFoundError(f"arquivo ausente ou vazio: {path}")
        sess = InferenceSession(path)
        MODEL_REGISTRY[key] = sess
        logger.info("%s carregado (%s).", label, path)
    except Exception as e:
        logger.warning("%s não disponível: %s", label, e)


def _load_onnx(key: str, path: str, label: str) -> None:
    try:
        import onnxruntime as ort
        MODEL_REGISTRY[key] = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        logger.info("%s carregado (%s).", label, path)
    except Exception as e:
        logger.warning("%s não disponível: %s", label, e)


def load_models() -> None:
    """Carrega todos os modelos ao iniciar. Falhas individuais não bloqueiam startup."""

    # ── Modelos legados ───────────────────────────────────────────────────────
    try:
        from models.chexpert_model import load_chexpert
        MODEL_REGISTRY["chexpert"] = load_chexpert()
        logger.info("CheXpert (densenet121-res224-all) carregado.")
    except Exception as e:
        logger.warning("CheXpert não disponível: %s", e)

    try:
        from models.ecg_model import load_ecg
        MODEL_REGISTRY["ecg"] = load_ecg()
        logger.info("ECG (rule-based clínico) carregado.")
    except Exception as e:
        logger.warning("ECG não disponível: %s", e)

    try:
        from models.derma_model import load_derma
        MODEL_REGISTRY["derma"] = load_derma()
        logger.info("HAM10000/EfficientNet-B4 carregado.")
    except Exception as e:
        logger.warning("Derma não disponível: %s", e)

    try:
        from models.sepse_model import _load_model
        _load_model()
        MODEL_REGISTRY["sepse"] = "xgboost-pkl-v1"
        logger.info("XGBoost Sepse (pkl, AUC 0.8766) carregado.")
    except Exception as e:
        logger.warning("Sepse não disponível: %s", e)

    try:
        from models.pill_model import load_pill
        MODEL_REGISTRY["pill"] = load_pill()
        logger.info("Pill Identifier (Claude Vision) carregado.")
    except Exception as e:
        logger.warning("Pill Identifier não disponível: %s", e)

    # ── Novos modelos tabular (ONNX) ─────────────────────────────────────────
    _load_pkl("cardiac",          "models/cardiac_xgboost_v2_combined.pkl",  "Cardiac XGBoost v2")
    _load_onnx_gbm("preeclampsia",  "models/preeclampsia_gbm.onnx",          "Preeclampsia GBM")
    _load_onnx_gbm("mortality",     "models/mortality_gbm.onnx",             "Mortality GBM")
    _load_onnx_gbm("readmissao",    "models/readmissao_gbm.onnx",            "Readmissão GBM")
    _load_onnx_gbm("deterioracao",  "models/deterioracao_gbm.onnx",          "Deterioração GBM")
    _load_onnx_gbm("vitaldb",       "models/vitaldb_ihi_v2.onnx",            "VitalDB IHI v2")
    _load_onnx_gbm("eeg",           "models/eeg_epilepsy_combined_gbm.onnx", "EEG Epilepsia GBM")
    _load_onnx_gbm("circor",        "models/circor_cardiac_gbm.onnx",        "CirCor Cardiac GBM")
    _load_onnx_gbm("lung_sound",    "models/lung_sound_ensemble.onnx",       "Lung Sound Ensemble")

    # ── Modelos tabular adicionais ───────────────────────────────────────
    _load_pkl("sepsis_v2",     "models/sepsis_xgboost.pkl",             "Sepse XGBoost v2 (AUC 0.8645)")
    _load_pkl("ecg_lgbm",      "models/ecg_lgbm_ptbxl.pkl",            "ECG LightGBM PTB-XL")

    # ── Modelos imagem PTH (EfficientNet-B0 / DenseNet-121) ────────────
    # Objetivo: registrar que foram baixados pelo download_models.py.
    # Inferencia real usa torch.load() diretamente no endpoint correspondente.
    _load_pth("skin_img",      "models/skin_efficientnet_b0_gpu.pth",   "Skin EfficientNet-B0")
    _load_pth("eyepacs",       "models/eyepacs_efficientnet_b0.pth",    "EyePACS EfficientNet-B0")
    _load_pth("chest_xray14",  "models/chest_xray_efficientnet_b0.pth", "Chest XR EfficientNet-B0")
    _load_pth("chestxray14",   "models/chestxray14_densenet121_v2.pth", "ChestX-ray14 DenseNet-121")
    _load_pth("brain_tumor",   "models/brain_tumor_efficientnet_b0.pth","Brain Tumor EfficientNet-B0")
    _load_pth("fractura",      "models/fractura_efficientnet_b0.pth",   "Fratura EfficientNet-B0")
    _load_pth("glaucoma_img",  "models/glaucoma_efficientnet_b0.pth",   "Glaucoma EfficientNet-B0")
    _load_pth("mamografia_img","models/mamografia_efficientnet_b0.pth", "Mamografia EfficientNet-B0")

    # BUG 3 FIX: registrar PTH paths sob as chaves onnx_* esperadas pelos endpoints de imagem
    _pth_img_map = {
        "onnx_chest":    "chest_xray14",
        "onnx_skin":     "skin_img",
        "onnx_retina":   "eyepacs",
        "onnx_brain":    "brain_tumor",
        "onnx_fracture": "fractura",
        "onnx_glaucoma": "glaucoma_img",
        "onnx_mammo":    "mamografia_img",
    }
    for onnx_key, pth_key in _pth_img_map.items():
        path = MODEL_REGISTRY.get(pth_key)
        if path:
            MODEL_REGISTRY[onnx_key] = path
            logger.info("Alias %s → %s registrado.", onnx_key, path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando download dos modelos...")
    try:
        import download_models  # executa o script de download
        logger.info("Download dos modelos concluído.")
    except Exception as e:
        logger.warning("Falha no download de modelos: %s", e)
    load_models()
    yield


app = FastAPI(
    title="Aldora AI Models",
    version="3.0.0",
    description=(
        "Microserviço de IA médica: 5 modelos legados + 11 tabular GBM/XGBoost + 7 ONNX imagem. "
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
    fc: float
    pr_ms: float
    qrs_ms: float
    qtc_ms: float
    rr_variability: float = 0.0
    paciente_id: Optional[str] = None


class ECGSignalsRequest(BaseModel):
    leads_100hz: list[list[float]]
    fc: Optional[float] = 0.0
    pr_ms: Optional[float] = 0.0
    qrs_ms: Optional[float] = 0.0
    qtc_ms: Optional[float] = 0.0
    rr_variability: Optional[float] = 0.0
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
    gender: Optional[float] = None
    unit1: Optional[float] = None
    unit2: Optional[float] = None
    hosp_adm_elapsed: Optional[float] = None
    icu_los_days: Optional[float] = None
    hora_no_icu: Optional[float] = None
    paciente_id: Optional[str] = None


# Novos schemas tabular
class CardiacRequest(BaseModel):
    murmur_presence: Optional[float] = 0.0   # 0=Absent, 1=Present
    age_years: Optional[float] = 30.0
    sex: Optional[float] = 0.0               # 0=F, 1=M
    pregnancy_status: Optional[float] = 0.0
    campaign: Optional[float] = 1.0


class PreeclampsiaRequest(BaseModel):
    age: Optional[float] = 25.0
    bmi: Optional[float] = 22.0
    systolic_bp: Optional[float] = 120.0
    diastolic_bp: Optional[float] = 80.0
    mean_arterial_pressure: Optional[float] = 93.0
    proteinuria: Optional[float] = 0.0
    creatinine: Optional[float] = 0.8
    platelets: Optional[float] = 250.0
    alt: Optional[float] = 20.0
    ast: Optional[float] = 20.0
    gestational_age: Optional[float] = 28.0


class MortalityRequest(BaseModel):
    age: Optional[float] = 50.0
    sofa_score: Optional[float] = 0.0
    apache_ii: Optional[float] = 0.0
    gcs: Optional[float] = 15.0
    lactate: Optional[float] = 1.0
    creatinine: Optional[float] = 0.8
    bilirubin: Optional[float] = 0.5
    platelets: Optional[float] = 250.0
    pao2_fio2: Optional[float] = 400.0
    vasopressors: Optional[float] = 0.0


class ReadmissaoRequest(BaseModel):
    age: Optional[float] = 50.0
    length_of_stay: Optional[float] = 5.0
    charlson_index: Optional[float] = 0.0
    lace_score: Optional[float] = 5.0
    previous_admissions: Optional[float] = 0.0
    diagnosis_icd: Optional[float] = 0.0   # codificado


class DeterioracaoRequest(BaseModel):
    news2_score: Optional[float] = 0.0
    delta_fc: Optional[float] = 0.0
    delta_fr: Optional[float] = 0.0
    delta_pas: Optional[float] = 0.0
    delta_temperatura: Optional[float] = 0.0
    delta_saturacao: Optional[float] = 0.0
    delta_glasgow: Optional[float] = 0.0


class VitalDBRequest(BaseModel):
    mbp: Optional[float] = 80.0
    hr: Optional[float] = 70.0
    spo2: Optional[float] = 98.0
    etco2: Optional[float] = 35.0
    bis: Optional[float] = 50.0
    temperature: Optional[float] = 36.5
    age: Optional[float] = 50.0
    asa_class: Optional[float] = 2.0
    operation_type: Optional[float] = 0.0


class EEGRequest(BaseModel):
    delta_power: Optional[float] = 0.0
    theta_power: Optional[float] = 0.0
    alpha_power: Optional[float] = 0.0
    beta_power: Optional[float] = 0.0
    gamma_power: Optional[float] = 0.0
    spectral_entropy: Optional[float] = 0.0
    hjorth_mobility: Optional[float] = 0.0
    hjorth_complexity: Optional[float] = 0.0


class CircorRequest(BaseModel):
    age_months: Optional[float] = 120.0
    weight: Optional[float] = 20.0
    height: Optional[float] = 110.0
    sex: Optional[float] = 0.0
    murmur_locations: Optional[float] = 0.0
    systolic_murmur_timing: Optional[float] = 0.0
    diastolic_murmur_timing: Optional[float] = 0.0


class LungSoundRequest(BaseModel):
    mfcc_mean: Optional[float] = 0.0
    mfcc_std: Optional[float] = 0.0
    spectral_centroid: Optional[float] = 0.0
    zero_crossing_rate: Optional[float] = 0.0
    rms_energy: Optional[float] = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos": {k: "carregado" for k in MODEL_REGISTRY},
        "total": len(MODEL_REGISTRY),
        "versao": "3.1.0",
        "timestamp": _ts(),
    }


# ── Legados ───────────────────────────────────────────────────────────────────

@app.post("/v1/chexpert")
async def chexpert(req: ImageRequest):
    """CheXpert — DenseNet121-res224-all. 14 patologias pulmonares."""
    modelo = MODEL_REGISTRY.get("chexpert")
    if modelo is None:
        raise HTTPException(503, "Modelo CheXpert não carregado.")
    try:
        resultado = modelo.predict(base64.b64decode(req.image_base64))
    except Exception as e:
        logger.exception("Erro CheXpert")
        raise HTTPException(500, f"Erro na inferência: {e}")
    top_conf = resultado[0]["probabilidade"] if resultado else 0.0
    return {"resultado": resultado, "confianca": round(top_conf, 4),
            "modelo": "torchxrayvision/densenet121-res224-all",
            "aviso": DISCLAIMER_CFM, "timestamp": _ts()}


@app.post("/v1/ecg")
async def ecg(req: ECGRequest):
    """ECG — análise rule-based clínica (ACC/AHA/SBC)."""
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
    return {"resultado": resultado, "confianca": round(top_conf, 4),
            "modelo": "rule-based-ecg-v1", "aviso": DISCLAIMER_CFM, "timestamp": _ts()}


@app.post("/ecg/analyze-signals")
async def analyze_ecg_signals(req: ECGSignalsRequest):
    """ECG com LightGBM PTB-XL a partir dos sinais brutos (12 derivações × 1000 amostras)."""
    modelo = MODEL_REGISTRY.get("ecg")
    if modelo is None:
        raise HTTPException(503, "Modelo ECG não carregado.")
    try:
        result = modelo.predict_from_signals(
            leads_100hz=req.leads_100hz,
            fc=req.fc or 0.0, pr_ms=req.pr_ms or 0.0,
            qrs_ms=req.qrs_ms or 0.0, qtc_ms=req.qtc_ms or 0.0,
            rr_variability=req.rr_variability or 0.0,
        )
    except Exception as e:
        logger.exception("Erro /ecg/analyze-signals")
        raise HTTPException(500, f"Erro na análise ECG-LightGBM: {e}")
    return {"resultado": result, "confianca": round(float(result.get("top_probabilidade", 0.0)), 4),
            "modelo": result.get("modelo", "lgbm_ptbxl"),
            "aviso": DISCLAIMER_CFM, "timestamp": _ts()}


@app.post("/v1/derma")
async def derma(req: ImageRequest):
    """HAM10000 — EfficientNet-B4. 7 classes de lesões cutâneas."""
    modelo = MODEL_REGISTRY.get("derma")
    if modelo is None:
        raise HTTPException(503, "Modelo Dermatologia não carregado.")
    try:
        resultado = modelo.predict(base64.b64decode(req.image_base64))
    except Exception as e:
        logger.exception("Erro Derma")
        raise HTTPException(500, f"Erro na inferência: {e}")
    top_conf = resultado[0]["probabilidade"] if resultado else 0.0
    return {"resultado": resultado, "confianca": round(top_conf, 4),
            "modelo": "efficientnet_b4-ham10000", "aviso": DISCLAIMER_CFM, "timestamp": _ts()}


@app.post("/v1/sepse")
async def sepse(req: SepseRequest):
    """XGBoost Sepse — PhysioNet 2019 + MIMIC-IV. AUC-ROC 0.8766."""
    from models.sepse_model import predict_sepsis
    try:
        resultado = predict_sepsis(_mapear_sepse_request(req))
    except Exception as e:
        logger.exception("Erro Sepse")
        raise HTTPException(500, f"Erro na inferência: {e}")
    return {"resultado": resultado, "confianca": resultado["probabilidade"],
            "modelo": "xgboost-sepse-physionet2019-pkl",
            "aviso": DISCLAIMER_CFM, "timestamp": _ts()}


@app.post("/sepsis/predict")
async def predict_sepsis_endpoint(request: Request):
    """XGBoost Sepse — endpoint direto com JSON livre. Aceita subconjunto das features."""
    try:
        dados = await request.json()
        # BUG 1 FIX: mapear campos PT-BR → EN (nomes das features do modelo XGBoost)
        _sepse_map = {
            'fc': 'HR', 'fr': 'Resp', 'temperatura': 'Temp',
            'pas': 'SBP', 'pad': 'DBP', 'spo2': 'O2Sat',
            'lactato': 'Lactate', 'leucocitos': 'WBC',
            'pcr': 'CRP', 'pct': 'PCT', 'glasgow': 'GCS',
            'ventilacao_mecanica': 'Mech_Vent', 'vasopressor': 'Vasopressor',
            'idade': 'Age', 'genero': 'Gender',
        }
        dados_en = {_sepse_map.get(k, k): v for k, v in dados.items()}
        from models.sepse_model import predict_sepsis
        resultado = predict_sepsis(dados_en)
        return {"success": True, "data": resultado}
    except Exception as e:
        logger.exception("Erro /sepsis/predict")
        return {"success": False, "error": str(e), "score": 50, "risco": "indeterminado"}


@app.post("/v1/pill")
async def pill(req: ImageRequest):
    """Pill Identifier — Claude Sonnet Vision."""
    modelo = MODEL_REGISTRY.get("pill")
    if modelo is None:
        raise HTTPException(503, "Pill Identifier não carregado. Verifique ANTHROPIC_API_KEY.")
    try:
        resultado = modelo.predict(base64.b64decode(req.image_base64), req.image_mime)
    except Exception as e:
        logger.exception("Erro Pill Identifier")
        raise HTTPException(500, f"Erro na identificação: {e}")
    nivel = resultado.get("confianca", "baixa")
    return {"resultado": resultado,
            "confianca": {"alta": 0.90, "media": 0.65, "baixa": 0.35}.get(nivel, 0.35),
            "modelo": "claude-sonnet-4-6-vision", "aviso": DISCLAIMER_CFM, "timestamp": _ts()}


# ── Novos endpoints tabular ───────────────────────────────────────────────────

@app.post("/cardiac/predict")
async def cardiac_predict(req: CardiacRequest):
    """
    Cardiac XGBoost v2 — CirCor 2022 + PCG combined.
    Features: murmur_presence, age_years, sex, pregnancy_status, campaign.
    """
    model = MODEL_REGISTRY.get("cardiac")
    if model is None:
        raise HTTPException(503, "Modelo cardiac_xgboost_v2_combined.pkl não carregado. Execute download_models.py.")
    try:
        features = ["murmur_presence", "age_years", "sex", "pregnancy_status", "campaign"]
        values = [req.murmur_presence, req.age_years, req.sex, req.pregnancy_status, req.campaign]
        return _tabular_infer(model, values, features, "cardiac_xgboost_v2_combined.pkl")
    except Exception as e:
        logger.exception("Erro /cardiac/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/preeclampsia/predict")
async def preeclampsia_predict(req: PreeclampsiaRequest):
    """
    Preeclampsia GBM — predição de pré-eclâmpsia.
    Features: age, bmi, systolic_bp, diastolic_bp, MAP, proteinuria,
    creatinine, platelets, alt, ast, gestational_age.
    """
    model = MODEL_REGISTRY.get("preeclampsia")
    if model is None:
        raise HTTPException(503, "Modelo preeclampsia_gbm.pkl não carregado. Execute download_models.py.")
    try:
        features = ["age", "bmi", "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
                    "proteinuria", "creatinine", "platelets", "alt", "ast", "gestational_age"]
        values = [req.age, req.bmi, req.systolic_bp, req.diastolic_bp, req.mean_arterial_pressure,
                  req.proteinuria, req.creatinine, req.platelets, req.alt, req.ast, req.gestational_age]
        return _tabular_infer(model, values, features, "preeclampsia_gbm.pkl")
    except Exception as e:
        logger.exception("Erro /preeclampsia/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/mortality/predict")
async def mortality_predict(req: MortalityRequest):
    """
    Mortality GBM — predição de mortalidade hospitalar (MIMIC-IV).
    Features: age, sofa_score, apache_ii, gcs, lactate, creatinine,
    bilirubin, platelets, pao2_fio2, vasopressors.
    """
    model = MODEL_REGISTRY.get("mortality")
    if model is None:
        raise HTTPException(503, "Modelo mortality_gbm.pkl não carregado. Execute download_models.py.")
    try:
        features = ["age", "sofa_score", "apache_ii", "gcs", "lactate",
                    "creatinine", "bilirubin", "platelets", "pao2_fio2", "vasopressors"]
        values = [req.age, req.sofa_score, req.apache_ii, req.gcs, req.lactate,
                  req.creatinine, req.bilirubin, req.platelets, req.pao2_fio2, req.vasopressors]
        return _tabular_infer(model, values, features, "mortality_gbm.pkl")
    except Exception as e:
        logger.exception("Erro /mortality/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/readmissao/predict")
async def readmissao_predict(req: ReadmissaoRequest):
    """
    Readmissão GBM — risco de readmissão em 30 dias.
    Features: age, length_of_stay, charlson_index, lace_score,
    previous_admissions, diagnosis_icd.
    """
    model = MODEL_REGISTRY.get("readmissao")
    if model is None:
        raise HTTPException(503, "Modelo readmissao_gbm.pkl não carregado. Execute download_models.py.")
    try:
        features = ["age", "length_of_stay", "charlson_index", "lace_score",
                    "previous_admissions", "diagnosis_icd"]
        values = [req.age, req.length_of_stay, req.charlson_index, req.lace_score,
                  req.previous_admissions, req.diagnosis_icd]
        return _tabular_infer(model, values, features, "readmissao_gbm.pkl")
    except Exception as e:
        logger.exception("Erro /readmissao/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/deterioracao/predict")
async def deterioracao_predict(req: DeterioracaoRequest):
    """
    Deterioração GBM — predição de deterioração clínica precoce (NEWS2 delta).
    Features: news2_score, delta_fc, delta_fr, delta_pas, delta_temperatura,
    delta_saturacao, delta_glasgow.
    """
    model = MODEL_REGISTRY.get("deterioracao")
    if model is None:
        raise HTTPException(503, "Modelo deterioracao_gbm.pkl não carregado. Execute download_models.py.")
    try:
        features = ["news2_score", "delta_fc", "delta_fr", "delta_pas",
                    "delta_temperatura", "delta_saturacao", "delta_glasgow"]
        values = [req.news2_score, req.delta_fc, req.delta_fr, req.delta_pas,
                  req.delta_temperatura, req.delta_saturacao, req.delta_glasgow]
        return _tabular_infer(model, values, features, "deterioracao_gbm.pkl")
    except Exception as e:
        logger.exception("Erro /deterioracao/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/vitaldb/predict")
async def vitaldb_predict(req: VitalDBRequest):
    """
    VitalDB IHI v2 — predição de eventos intraoperatórios (VitalDB 2024).
    Features: mbp, hr, spo2, etco2, bis, temperature, age, asa_class, operation_type.
    """
    model = MODEL_REGISTRY.get("vitaldb")
    if model is None:
        raise HTTPException(503, "Modelo vitaldb_ihi_v2.pkl não carregado. Execute download_models.py.")
    try:
        features = ["mbp", "hr", "spo2", "etco2", "bis", "temperature",
                    "age", "asa_class", "operation_type"]
        values = [req.mbp, req.hr, req.spo2, req.etco2, req.bis, req.temperature,
                  req.age, req.asa_class, req.operation_type]
        return _tabular_infer(model, values, features, "vitaldb_ihi_v2.pkl")
    except Exception as e:
        logger.exception("Erro /vitaldb/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/eeg/predict")
async def eeg_predict(req: EEGRequest):
    """
    EEG Epilepsia GBM — classificação de atividade epileptiforme por banda espectral.
    Features: delta_power, theta_power, alpha_power, beta_power, gamma_power,
    spectral_entropy, hjorth_mobility, hjorth_complexity.
    """
    model = MODEL_REGISTRY.get("eeg")
    if model is None:
        raise HTTPException(503, "Modelo eeg_epilepsy_combined_gbm.pkl não carregado. Execute download_models.py.")
    try:
        features = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
                    "spectral_entropy", "hjorth_mobility", "hjorth_complexity"]
        values = [req.delta_power, req.theta_power, req.alpha_power, req.beta_power, req.gamma_power,
                  req.spectral_entropy, req.hjorth_mobility, req.hjorth_complexity]
        return _tabular_infer(model, values, features, "eeg_epilepsy_combined_gbm.pkl")
    except Exception as e:
        logger.exception("Erro /eeg/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/circor/predict")
async def circor_predict(req: CircorRequest):
    """
    CirCor Cardiac GBM — detecção de sopro cardíaco pediátrico (CirCor 2022).
    Features: age_months, weight, height, sex, murmur_locations,
    systolic_murmur_timing, diastolic_murmur_timing.
    """
    model = MODEL_REGISTRY.get("circor")
    if model is None:
        raise HTTPException(503, "Modelo circor_cardiac_gbm.pkl não carregado. Execute download_models.py.")
    try:
        features = ["age_months", "weight", "height", "sex", "murmur_locations",
                    "systolic_murmur_timing", "diastolic_murmur_timing"]
        values = [req.age_months, req.weight, req.height, req.sex, req.murmur_locations,
                  req.systolic_murmur_timing, req.diastolic_murmur_timing]
        return _tabular_infer(model, values, features, "circor_cardiac_gbm.pkl")
    except Exception as e:
        logger.exception("Erro /circor/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/lung/predict")
async def lung_predict(req: LungSoundRequest):
    """
    Lung Sound Ensemble — classificação de sons respiratórios (ICBHI 2017).
    Features: mfcc_mean, mfcc_std, spectral_centroid, zero_crossing_rate, rms_energy.
    """
    model = MODEL_REGISTRY.get("lung_sound")
    if model is None:
        raise HTTPException(503, "Modelo lung_sound_ensemble.pkl não carregado. Execute download_models.py.")
    try:
        features = ["mfcc_mean", "mfcc_std", "spectral_centroid", "zero_crossing_rate", "rms_energy"]
        values = [req.mfcc_mean, req.mfcc_std, req.spectral_centroid, req.zero_crossing_rate, req.rms_energy]
        return _tabular_infer(model, values, features, "lung_sound_ensemble.pkl")
    except Exception as e:
        logger.exception("Erro /lung/predict")
        raise HTTPException(500, f"Erro na inferência: {e}")


# ── Novos endpoints ONNX (imagem) ─────────────────────────────────────────────

@app.post("/image/chest-xray")
async def image_chest_xray(image: UploadFile = File(...)):
    """
    Chest X-Ray ONNX — EfficientNet-B0. Classificação: normal / anormal.
    Entrada: multipart/form-data campo 'image' (JPEG/PNG).
    """
    session = MODEL_REGISTRY.get("onnx_chest")
    if session is None:
        raise HTTPException(503, "Modelo chest_xray.onnx não carregado. Execute download_models.py.")
    try:
        return _onnx_infer(session, await image.read(), _ONNX_LABELS["chest_xray"], "chest_xray.onnx")
    except Exception as e:
        logger.exception("Erro /image/chest-xray")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/image/skin")
async def image_skin(image: UploadFile = File(...)):
    """
    Skin Dermatologia ONNX — EfficientNet-B0. 7 classes: nv, mel, bkl, bcc, akiec, vasc, df.
    Entrada: multipart/form-data campo 'image' (JPEG/PNG).
    """
    session = MODEL_REGISTRY.get("onnx_skin")
    if session is None:
        raise HTTPException(503, "Modelo skin_dermatologia.onnx não carregado. Execute download_models.py.")
    try:
        return _onnx_infer(session, await image.read(), _ONNX_LABELS["skin"], "skin_dermatologia.onnx")
    except Exception as e:
        logger.exception("Erro /image/skin")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/image/retinopathy")
async def image_retinopathy(image: UploadFile = File(...)):
    """
    Retinopatia ONNX — EfficientNet-B0. Graduação: grau_0 (sem) a grau_4 (proliferativo).
    Entrada: multipart/form-data campo 'image' (JPEG/PNG).
    """
    session = MODEL_REGISTRY.get("onnx_retina")
    if session is None:
        raise HTTPException(503, "Modelo retinopatia.onnx não carregado. Execute download_models.py.")
    try:
        return _onnx_infer(session, await image.read(), _ONNX_LABELS["retinopathy"], "retinopatia.onnx")
    except Exception as e:
        logger.exception("Erro /image/retinopathy")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/image/brain-tumor")
async def image_brain_tumor(image: UploadFile = File(...)):
    """
    Hemorragia Cerebral ONNX — EfficientNet-B0. Classes: glioma, meningioma, pituitary, no_tumor.
    Entrada: multipart/form-data campo 'image' (JPEG/PNG).
    """
    session = MODEL_REGISTRY.get("onnx_brain")
    if session is None:
        raise HTTPException(503, "Modelo hemorragia_cerebral.onnx não carregado. Execute download_models.py.")
    try:
        return _onnx_infer(session, await image.read(), _ONNX_LABELS["brain_tumor"], "hemorragia_cerebral.onnx")
    except Exception as e:
        logger.exception("Erro /image/brain-tumor")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/image/fracture")
async def image_fracture(image: UploadFile = File(...)):
    """
    Fratura Óssea ONNX — EfficientNet-B0. Classificação: normal / fratura.
    Entrada: multipart/form-data campo 'image' (JPEG/PNG).
    """
    session = MODEL_REGISTRY.get("onnx_fracture")
    if session is None:
        raise HTTPException(503, "Modelo fratura_ossea.onnx não carregado. Execute download_models.py.")
    try:
        return _onnx_infer(session, await image.read(), _ONNX_LABELS["fracture"], "fratura_ossea.onnx")
    except Exception as e:
        logger.exception("Erro /image/fracture")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/image/glaucoma")
async def image_glaucoma(image: UploadFile = File(...)):
    """
    Glaucoma ONNX — EfficientNet-B0. Classificação: normal / glaucoma.
    Entrada: multipart/form-data campo 'image' (JPEG/PNG).
    """
    session = MODEL_REGISTRY.get("onnx_glaucoma")
    if session is None:
        raise HTTPException(503, "Modelo glaucoma.onnx não carregado. Execute download_models.py.")
    try:
        return _onnx_infer(session, await image.read(), _ONNX_LABELS["glaucoma"], "glaucoma.onnx")
    except Exception as e:
        logger.exception("Erro /image/glaucoma")
        raise HTTPException(500, f"Erro na inferência: {e}")


@app.post("/image/mammography")
async def image_mammography(image: UploadFile = File(...)):
    """
    Mamografia ONNX — EfficientNet-B0. Classificação: normal / anormal.
    Entrada: multipart/form-data campo 'image' (JPEG/PNG).
    """
    session = MODEL_REGISTRY.get("onnx_mammo")
    if session is None:
        raise HTTPException(503, "Modelo mamografia.onnx não carregado. Execute download_models.py.")
    try:
        return _onnx_infer(session, await image.read(), _ONNX_LABELS["mammography"], "mamografia.onnx")
    except Exception as e:
        logger.exception("Erro /image/mammography")
        raise HTTPException(500, f"Erro na inferência: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mapear_sepse_request(req: SepseRequest) -> dict:
    """Mapeia SepseRequest → dict com feature names do modelo XGBoost."""
    return {
        'HR': req.hr, 'O2Sat': req.o2sat, 'Temp': req.temp,
        'SBP': req.sbp, 'MAP': req.map, 'DBP': req.dbp, 'Resp': req.resp,
        'BUN': req.bun, 'Glucose': req.glucose, 'Lactate': req.lactate,
        'Potassium': req.potassium, 'Creatinine': req.creatinine,
        'Hct': req.hct, 'Hgb': req.hgb, 'WBC': req.wbc,
        'Platelets': req.platelets, 'Age': req.age, 'Gender': req.gender,
        'ICULOS': req.icu_los_days,
    }
