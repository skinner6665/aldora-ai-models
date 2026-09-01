"""
Aldora AI Models — Modal.com GPU Deployment v4.0.0
Migração Railway → Modal A10G
Latência imagem: 2-5s (vs 50-110s Railway CPU)
Endpoints: mesmos do main.py — interface REST idêntica
CFM 2.454/2026 — apoio diagnóstico, não substituição clínica.
"""
import io
import os
import sys
import base64
import logging
import pickle

import numpy as np
import modal

# ─── Modal App ────────────────────────────────────────────────────────────────

app = modal.App("aldora-ai-models")

MODEL_DIR = "/models"
volume = modal.Volume.from_name("aldora-models-v1", create_if_missing=True)

# Modelos grandes → Google Drive → Volume (download na primeira inicialização)
DRIVE_MODELS = [
    # Tabular PKL/ONNX (não presentes localmente)
    ("cardiac_xgboost_v2_combined.pkl", "1BrbaUR-c0mJl8n3flJA-o7VE497Hengl"),
    ("risco_materno_gbm.onnx",          "10WAr9OoV_HNSvGEMoeXjJvwK108BIjWa"),
    ("mortality_xgb_v2.json",         "1-jrXeNuxirJLXj3lHhrFC0dJPeTsupUv"),
    ("mortality_xgb_v2_colunas.json", "1KpfN0dinRDJQ95CkOz3Mwpd-XB-HHMG5"),
    ("readmissao_gbm.onnx",             "1CjHBGeCm44LanRMUCWWi7kT2oNYmquqC"),
    ("deterioracao_gbm.onnx",           "1CMg9igOO1Lg0d-_C-hjNeFXGN9idSgcT"),
    ("vitaldb_ihi_v2.onnx",             "1rp03ROSuaOzQSuckVwojq-5pJqoHGDMG"),
    ("eeg_epilepsy_combined_gbm.onnx",  "14NZlNndyoWoQtkaKNQTrO6-Etbopl5Oy"),
    ("circor_cardiac_gbm.onnx",         "1e45tncqP_2wAm9u_M5E07Y2TAasZaKkd"),
    ("lung_sound_ensemble.onnx",        "1kZjWbcNtdT3Tl7l3iXZWKMPNPP79OvA6"),
    # Brain Tumor v2 ONNX binário sigmoid — pesos embutidos (16.5MB, AUC=0.9995)
    ("brain_tumor_v2_combined.onnx",   "1g-8rDCroAgSzSKbojwnN6qj9IK0cXSD9"),
    ("tc_cranio_v4_ct.onnx",           "1fLbsYf8wRH3enpMRDbaiqfEef6uohWZw"),
    ("mamografia_v2_busi.onnx",        "1eeVIgjQHp9znLONF16d3G0-Sd856F6ed"),
    # Imagem — EfficientNet-B0 PTH
    ("skin_efficientnet_b0_gpu.pth",    "1djuI95JgSJt7nJ71mxFJsSLEU_cTljUI"),
    ("eyepacs_efficientnet_b0.pth",     "1Ja7DSuWfck-v397bAPAUHyTS9XlWhqzN"),
    ("chest_xray_efficientnet_b0.pth",  "16O2beLazd8pXAKb6hizqEUSllppnilxE"),
    ("brain_tumor_efficientnet_b0.pth", "1KpZgTrJmCCuEFPvMJ1DtJLDm4RPWL6Pd"),
    ("fractura_efficientnet_b0.pth",    "1IINhfX72E3dNTegn-P_rcUDNHAFsUkGj"),
    ("glaucoma_efficientnet_b0.pth",    "18-IrMNiiFn7YYWTsA1AkIwGk5DAOMuBa"),
    ("mamografia_v5_best.pth",          "1y1AqJV2286JQIeGKaUoqRlIBRbnKjG-L"),
    ("chestxray14_densenet121_v2.pth",  "14Zc7kRQB07JEGe9A6wqDpl92xNKKBq84"),
    # ECG CODE-15% (Fase B) — Pesos PyTorch (ResNet1D)
    # IDs confirmados via links do Drive fornecidos pelo usuário
    ("code15_faseB18_fold0.pt",         "1CK0WZTegzEPQEeukJ4u2US07_-Y3unFa"),
    ("code15_faseB18_fold1.pt",         "1toiMk9bUY_oCuiLynk3WGSRE7dw-Sj9c"),
    ("code15_faseB18_fold2.pt",         "1r7z90NDSkSacbJqmv3RASGHMTaJ0rilD"),
    ("code15_faseB18_fold3.pt",         "1HPvIu8WOcUXeGXEcAPXXqk9lAGSzWUDS"),
    ("code15_faseB18_fold4.pt",         "1yBV2tZzRbroYA7jc4mNaXKGv0l8NRgzE"),
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1", "libglib2.0-0", "libgomp1", "libsm6", "libxext6"])
    .pip_install([
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.9",
        # PyTorch — versão com CUDA
        "torch==2.2.2",
        "torchvision==0.17.2",
        "timm>=1.0.0",
        "torchxrayvision>=1.0.1",
        # ONNX Runtime com GPU
        "onnxruntime-gpu>=1.18.0",
        "numpy>=1.26.0,<2.0",
        # Tabular
        "xgboost>=2.0.0",
        "scikit-learn>=1.4.0,<1.7",
        "pandas>=2.2.0",
        "joblib>=1.4.0",
        "lightgbm==4.6.0",
        "scipy>=1.10.0",
        # Download / Pill
        "gdown>=5.2.0",
        "anthropic>=0.40.0",
        "Pillow>=10.0.0",
    ])
    # Copia arquivos Python do projeto + PKL pequenos (já existem localmente)
    .add_local_dir("models", remote_path="/app/models")
)

# ─── Constantes ───────────────────────────────────────────────────────────────

DISCLAIMER_CFM = (
    "AVISO REGULATÓRIO — CFM Resolução 2.454/2026: Este resultado é gerado por sistema de "
    "apoio diagnóstico (SaMD/ANVISA). NÃO substitui avaliação clínica por profissional habilitado. "
    "Não comunicar ao paciente sem mediação do médico responsável."
)
DISCLAIMER_SHORT = "Ferramenta de apoio à decisão clínica. Decisão final é do médico. CFM 2.454/2026."

# Labels ONNX — ordem dos neurônios de saída (hardcoded, não depende dos PKL)
_ONNX_LABELS: dict[str, list[str]] = {
    "chest_xray":  ["normal", "anormal"],
    "skin":        ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"],
    "retinopathy": ["grau_0", "grau_1", "grau_2", "grau_3", "grau_4"],
    "brain_tumor": ["glioma", "meningioma", "pituitary", "no_tumor"],
    "fracture":    ["normal", "fratura"],
    "glaucoma":    ["normal", "glaucoma"],
    "mammography": ["normal", "suspeito"],
}

try:
    import torchvision.transforms as _T
    _norm_mammo = _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    _MAMOGRAFIA_V5_TTA = [
        _T.Compose([_T.Resize((224, 224)), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((224, 224)), _T.RandomHorizontalFlip(p=1.0), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((256, 256)), _T.CenterCrop(224), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((224, 224)), _T.RandomVerticalFlip(p=1.0), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((256, 256)), _T.CenterCrop(224), _T.RandomHorizontalFlip(p=1.0), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((240, 240)), _T.CenterCrop(224), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((240, 240)), _T.CenterCrop(224), _T.RandomHorizontalFlip(p=1.0), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((224, 224)), _T.RandomRotation(10), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((256, 256)), _T.CenterCrop(224), _T.RandomVerticalFlip(p=1.0), _T.ToTensor(), _norm_mammo]),
        _T.Compose([_T.Resize((280, 280)), _T.CenterCrop(224), _T.ToTensor(), _norm_mammo]),
    ]
except ImportError:
    _MAMOGRAFIA_V5_TTA = []

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _preprocess_onnx(image_bytes: bytes) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
        / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


_PTH_CACHE: dict = {}

def _onnx_infer(session, image_bytes: bytes, labels: list[str], model_file: str) -> dict:
    inp = _preprocess_onnx(image_bytes)
    if isinstance(session, str):
        # PTH path — EfficientNet-B0 com GPU
        import torch
        if session not in _PTH_CACHE:
            checkpoint = torch.load(session, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                from torchvision.models import efficientnet_b0
                model = efficientnet_b0(weights=None)
                in_f = model.classifier[1].in_features
                model.classifier[1] = torch.nn.Linear(in_f, len(labels))
                state = {k.replace("module.", ""): v for k, v in checkpoint.items()}
                model.load_state_dict(state, strict=False)
            else:
                model = checkpoint
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            _PTH_CACHE[session] = model
        m = _PTH_CACHE[session]
        device = next(m.parameters()).device
        t = torch.tensor(inp).to(device)
        with torch.no_grad():
            out = torch.softmax(m(t), dim=1)[0].cpu().numpy()
    else:
        out = session.run(None, {session.get_inputs()[0].name: inp})[0][0]
        if abs(float(out.sum()) - 1.0) > 0.05:
            out = _softmax(out)

    idx = int(out.argmax())
    return {
        "predicao":      labels[idx],
        "confianca":     round(float(out[idx]), 4),
        "scores":        {labels[i]: round(float(out[i]), 4) for i in range(len(labels))},
        "modelo_versao": model_file,
        "disclaimer":    DISCLAIMER_SHORT,
    }


def _mamografia_v5_tta_infer(model_path: str, image_bytes: bytes) -> dict:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import models as tv_models

    if model_path not in _PTH_CACHE:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = tv_models.efficientnet_b4(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f, 2),
        )
        state = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        _PTH_CACHE[model_path] = model

    m = _PTH_CACHE[model_path]
    device = next(m.parameters()).device
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tta_probs = []
    with torch.no_grad():
        for tfm in _MAMOGRAFIA_V5_TTA:
            t = tfm(img).unsqueeze(0).to(device)
            prob = torch.softmax(m(t), dim=1)[0, 1].item()
            tta_probs.append(prob)

    prob_suspeito = float(np.mean(tta_probs))
    prob_normal   = 1.0 - prob_suspeito
    label = "suspeito" if prob_suspeito >= 0.5 else "normal"

    return {
        "resultado":       label,
        "confianca":       round(max(prob_suspeito, prob_normal) * 100, 1),
        "probabilidades":  {
            "normal":   round(prob_normal   * 100, 1),
            "suspeito": round(prob_suspeito * 100, 1),
        },
        "tta_transforms":  len(_MAMOGRAFIA_V5_TTA),
        "auc_validacao":   0.8654,
        "modelo":          "EfficientNet-B4 TTA-10",
        "aviso":           DISCLAIMER_CFM,
        "timestamp":       _ts(),
    }


def _tabular_infer(model, values: list, features: list[str], model_file: str) -> dict:
    try:
        from onnxruntime import InferenceSession as _OrtSession
        if isinstance(model, _OrtSession):
            expected_shape = model.get_inputs()[0].shape
            if len(expected_shape) > 1 and isinstance(expected_shape[1], int):
                if expected_shape[1] != len(values):
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=400,
                        detail=f"Shape mismatch for {model_file}: model expects {expected_shape[1]} features, got {len(values)}"
                    )
            arr = np.array([values], dtype=np.float32)
            outputs = model.run(None, {model.get_inputs()[0].name: arr})
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
                "predicao": label, "confianca": round(max(p0, p1), 4),
                "scores": {"0": round(p0, 4), "1": round(p1, 4)},
                "modelo_versao": model_file, "disclaimer": DISCLAIMER_SHORT,
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
            "predicao": classes[idx], "confianca": round(float(proba[idx]), 4),
            "scores": {c: round(float(p), 4) for c, p in zip(classes, proba)},
            "modelo_versao": model_file, "disclaimer": DISCLAIMER_SHORT,
        }
    pred = float(model.predict(X)[0])
    label = "1" if pred >= 0.5 else "0"
    return {
        "predicao": label, "confianca": round(abs(pred - 0.5) + 0.5, 4),
        "scores": {"0": round(1.0 - pred, 4), "1": round(pred, 4)},
        "modelo_versao": model_file, "disclaimer": DISCLAIMER_SHORT,
    }


# ─── Sepse: score híbrido XGBoost + qSOFA/SIRS (v1.1.0) ────────────────────

_SEPSE_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "BUN", "Glucose", "Lactate", "Potassium", "Creatinine",
    "Hct", "Hgb", "WBC", "Platelets", "Age", "Gender", "ICULOS",
]
_SEPSE_MEDIANS: dict = {
    "HR": 83.0, "O2Sat": 97.0, "Temp": 36.9, "SBP": 122.0,
    "MAP": 82.0, "DBP": 63.0, "Resp": 18.0, "BUN": 18.0,
    "Glucose": 128.0, "Lactate": 1.6, "Potassium": 4.0,
    "Creatinine": 0.9, "Hct": 31.6, "Hgb": 10.7,
    "WBC": 10.7, "Platelets": 213.0, "Age": 62.0, "Gender": 1.0, "ICULOS": 24.0,
}
_SEPSE_PT_MAP = {
    "fc": "HR", "fr": "Resp", "temperatura": "Temp",
    "pas": "SBP", "pad": "DBP", "spo2": "O2Sat",
    "lactato": "Lactate", "leucocitos": "WBC",
    "pcr": "CRP", "pct": "PCT", "glasgow": "GCS",
    "ventilacao_mecanica": "Mech_Vent", "vasopressor": "Vasopressor",
    "idade": "Age", "genero": "Gender",
    "horas_uti": "ICULOS",
}
_sepse_xgb_model = None

def _load_sepse_xgb():
    global _sepse_xgb_model
    if _sepse_xgb_model is None:
        path = "/app/models/sepsis_xgboost.pkl"
        with open(path, "rb") as f:
            _sepse_xgb_model = pickle.load(f)["model"]
    return _sepse_xgb_model


def _clinical_score_sepse(dados: dict) -> tuple:
    qsofa = 0
    gcs = dados.get("GCS")
    resp = dados.get("Resp")
    sbp = dados.get("SBP")
    hr = dados.get("HR")
    temp = dados.get("Temp")
    wbc = dados.get("WBC")
    lactate = dados.get("Lactate")
    map_val = dados.get("MAP")
    spo2 = dados.get("O2Sat")

    if gcs is not None and float(gcs) < 15:
        qsofa += 1
    if resp is not None and float(resp) >= 22:
        qsofa += 1
    if sbp is not None and float(sbp) <= 100:
        qsofa += 1

    sirs = 0
    if temp is not None:
        t = float(temp)
        if t > 38.3 or t < 36.0:
            sirs += 1
    if hr is not None and float(hr) > 90:
        sirs += 1
    if resp is not None and float(resp) > 20:
        sirs += 1
    if wbc is not None:
        w = float(wbc)
        if w > 12.0 or w < 4.0:
            sirs += 1

    lactato_comp = 0.0
    if lactate is not None and float(lactate) > 2.0:
        lactato_comp = min(1.0, (float(lactate) - 2.0) / 4.0)

    map_comp = 0.0
    if map_val is not None and float(map_val) < 65.0:
        map_comp = min(1.0, (65.0 - float(map_val)) / 30.0)

    spo2_comp = 0.0
    if spo2 is not None and float(spo2) < 94.0:
        spo2_comp = min(1.0, (94.0 - float(spo2)) / 10.0)

    clinical_prob = (
        (qsofa / 3.0) * 0.35 +
        (sirs / 4.0) * 0.20 +
        lactato_comp * 0.20 +
        map_comp * 0.15 +
        spo2_comp * 0.10
    )
    detalhes = {
        "qsofa": qsofa, "sirs": sirs,
        "lactato_critico": lactate is not None and float(lactate) > 2.0,
        "map_choque": map_val is not None and float(map_val) < 65.0,
        "hipoxemia": spo2 is not None and float(spo2) < 94.0,
    }
    return clinical_prob, detalhes


def predict_sepsis_hybrid(dados_en: dict) -> dict:
    dados = dict(dados_en)
    if dados.get("MAP") is None and dados.get("SBP") is not None and dados.get("DBP") is not None:
        dados["MAP"] = round((float(dados["SBP"]) + 2 * float(dados["DBP"])) / 3, 1)
    if dados.get("ICULOS") is None:
        dados["ICULOS"] = 0.0

    model = _load_sepse_xgb()
    X = np.array([[
        float(dados[f] if dados.get(f) is not None else _SEPSE_MEDIANS[f])
        for f in _SEPSE_FEATURES
    ]])
    xgb_prob = float(model.predict_proba(X)[0][1])

    clinical_prob, detalhes = _clinical_score_sepse(dados)
    blended = xgb_prob * 0.40 + clinical_prob * 0.60
    score = round(blended * 100, 1)

    if blended < 0.10:
        risco, cor, msg = "baixo", "green", "Baixo risco de sepse"
    elif blended < 0.25:
        risco, cor, msg = "moderado", "yellow", "Risco moderado — monitorar sinais vitais e lactato"
    elif blended < 0.50:
        risco, cor, msg = "alto", "orange", "Alto risco — considerar avaliacao imediata e culturas"
    else:
        risco, cor, msg = "critico", "red", "Risco critico — protocolo de sepse recomendado"

    features_ok = [f for f in _SEPSE_FEATURES if dados.get(f) is not None]
    features_imp = [f for f in _SEPSE_FEATURES if dados.get(f) is None]

    return {
        "score": score, "probabilidade": blended, "risco": risco, "cor": cor,
        "mensagem": msg, "features_utilizadas": len(features_ok),
        "features_imputadas": features_imp,
        "componentes": {"xgboost_score": round(xgb_prob * 100, 1), **detalhes},
        "auc_modelo": 0.8766, "versao_modelo": "1.1.0-aldora-blend-xgb-qsofa",
        "dataset_treino": "PhysioNet2019 + MIMIC-IV + eICU",
        "disclaimer": DISCLAIMER_SHORT,
    }


# ─── AldoraAI Modal Class ─────────────────────────────────────────────────────

@app.cls(
    gpu="A10G",
    volumes={MODEL_DIR: volume},
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("aldora-secrets")],
)
class AldoraAI:

    @modal.enter()
    def startup(self) -> None:
        """Executa na inicialização do container: download + carregamento de modelos."""
        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger("aldora-modal")

        # Garante que /app/models está no path Python
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

        # Torna o diretório de modelos persistente e cria subdiretórios
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Cache torchxrayvision → Volume (evita re-download a cada cold start)
        os.environ["TORCH_HOME"] = MODEL_DIR

        self._download_drive_models()
        self._create_label_pkls()
        self._load_all_models()
        self._log.info("AldoraAI pronto. Modelos: %d", len(self.registry))

    def _download_drive_models(self) -> None:
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown -q")
            import gdown

        errors = []
        for fname, fid in DRIVE_MODELS:
            dest = f"{MODEL_DIR}/{fname}"
            if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
                self._log.info("[SKIP] %s", fname)
                continue
            url = f"https://drive.google.com/uc?id={fid}"
            self._log.info("[DOWN] %s ...", fname)
            try:
                gdown.download(url, dest, quiet=True)
                if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
                    self._log.info("[OK] %s (%.1f MB)", fname, os.path.getsize(dest) / 1e6)
                else:
                    raise FileNotFoundError("arquivo vazio ou muito pequeno após download")
            except Exception as e:
                self._log.warning("[ERR] %s: %s", fname, e)
                errors.append(fname)
        if errors:
            self._log.warning("Falhas no download: %s — endpoints afetados retornam 503", errors)
        volume.commit()

    def _create_label_pkls(self) -> None:
        """Recria PKLs de label encoders no Volume se ausentes."""
        encoders = {
            "chest_xray_classes.pkl":    ["anormal", "normal"],
            "brain_tumor_classes.pkl":   ["glioma", "meningioma", "notumor", "pituitary"],
            "fractura_classes.pkl":      ["fractured", "normal"],
            "glaucoma_classes.pkl":      ["glaucoma", "normal"],
            "mamografia_classes.pkl":    ["anormal", "normal"],
            "skin_label_encoder_gpu.pkl":["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"],
            "eyepacs_label_encoder.pkl": ["grau_0", "grau_1", "grau_2", "grau_3", "grau_4"],
        }
        for fname, classes in encoders.items():
            path = f"{MODEL_DIR}/{fname}"
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    pickle.dump(classes, f)
        volume.commit()

    def _load_pth(self, key: str, fname: str, label: str) -> None:
        path = f"{MODEL_DIR}/{fname}"
        try:
            if not os.path.exists(path) or os.path.getsize(path) < 100_000:
                raise FileNotFoundError(f"ausente ou vazio: {path}")
            self.registry[key] = path
            self._log.info("%s registrado (%s).", label, fname)
        except Exception as e:
            self._log.warning("%s indisponível: %s", label, e)

    def _load_onnx(self, key: str, fname: str, label: str) -> None:
        path = f"{MODEL_DIR}/{fname}"
        try:
            import onnxruntime as ort
            if not os.path.exists(path) or os.path.getsize(path) < 1000:
                raise FileNotFoundError(f"ausente ou vazio: {path}")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.registry[key] = ort.InferenceSession(path, providers=providers)
            self._log.info("%s carregado (%s).", label, fname)
        except Exception as e:
            self._log.warning("%s indisponível: %s", label, e)

    def _load_onnx_local(self, key: str, fname: str, label: str, min_mb: float = 1.0) -> None:
        """Carrega ONNX de /app/models (bundled no image Docker)."""
        path = f"/app/models/{fname}"
        try:
            import onnxruntime as ort
            if not os.path.exists(path):
                raise FileNotFoundError(f"ONNX ausente: {path}")
            size_mb = os.path.getsize(path) / 1e6
            if size_mb < min_mb:
                raise ValueError(f"ONNX muito pequeno ({size_mb:.1f}MB) — verificar arquivo")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(path, providers=providers)
            self.registry[key] = sess
            self._log.info("%s carregado (%s, %.1fMB).", label, fname, size_mb)
        except Exception as e:
            self._log.warning("%s indisponível: %s", label, e)

    def _load_brain_v2(self) -> None:
        """Carrega brain_tumor_v2_combined.onnx (binário sigmoid, AUC=0.9995, pesos embutidos 16.5MB).
        Se ausente, fallback para o PTH 4-classes."""
        onnx_path = f"{MODEL_DIR}/brain_tumor_v2_combined.onnx"
        try:
            import onnxruntime as ort
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX ausente: {onnx_path}")
            size_mb = os.path.getsize(onnx_path) / 1e6
            if size_mb < 10:
                raise ValueError(f"ONNX muito pequeno ({size_mb:.1f}MB) — versão antiga sem pesos embutidos")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            self.registry["onnx_brain_v2"] = sess
            self._log.info("Brain Tumor v2 ONNX binário carregado (AUC=0.9995, %.1fMB).", size_mb)
        except Exception as e:
            self._log.warning("Brain Tumor v2 indisponível: %s — usando PTH 4-classes.", e)

    def _load_pkl(self, key: str, fname: str, label: str, volume_path: bool = True) -> None:
        path = f"{MODEL_DIR}/{fname}" if volume_path else f"/app/models/{fname}"
        try:
            import joblib
            if not os.path.exists(path) or os.path.getsize(path) < 100:
                raise FileNotFoundError(f"ausente: {path}")
            self.registry[key] = joblib.load(path)
            self._log.info("%s carregado (%s).", label, fname)
        except Exception as e:
            self._log.warning("%s indisponível: %s", label, e)

    def _load_xgb_native(self, key: str, model_fname: str, cols_fname: str, label: str, auc: float, n: int) -> None:
        """Carrega booster XGBoost nativo (JSON) e lista de colunas."""
        model_path = f"{MODEL_DIR}/{model_fname}"
        cols_path = f"{MODEL_DIR}/{cols_fname}"
        try:
            import xgboost as xgb
            import json
            if not os.path.exists(model_path) or not os.path.exists(cols_path):
                raise FileNotFoundError(f"arquivos ausentes: {model_path} ou {cols_path}")
            b = xgb.Booster()
            b.load_model(model_path)
            with open(cols_path, "r") as f:
                contrato = json.load(f)
            colunas = contrato["colunas"]
            if len(colunas) != contrato.get("n", len(colunas)):
                raise ValueError(f"contrato inconsistente: {len(colunas)} colunas, n={contrato.get('n')}")
            self.registry[key] = {"booster": b, "colunas": colunas, "auc": auc, "n": n}
            self._log.info("%s carregado (%d colunas, AUC %.4f).", label, len(colunas), auc)
        except Exception as e:
            self._log.warning("%s indisponível: %s", label, e)

    def _load_all_models(self) -> None:
        self.registry: dict = {}

        # CheXpert — DenseNet121 (torchxrayvision auto-download)
        try:
            from models.chexpert_model import load_chexpert
            self.registry["chexpert"] = load_chexpert()
            self._log.info("CheXpert carregado.")
        except Exception as e:
            self._log.warning("CheXpert indisponível: %s", e)

        # ECG — rule-based + LightGBM PTB-XL (Legado)
        try:
            from models.ecg_model import load_ecg
            self.registry["ecg"] = load_ecg()
            self._log.info("ECG LightGBM PTB-XL carregado.")
        except Exception as e:
            self._log.warning("ECG indisponível: %s", e)

        # ECG — CODE-15% (Novo, Brasileiro, ResNet1D Ensemble)
        try:
            from models.ecg_code15 import load_ecg_code15
            predictor = load_ecg_code15()
            if predictor.loaded:
                self.registry["ecg_code15"] = predictor
                self._log.info("ECG CODE-15% carregado (%d folds).", len(predictor.models))
            else:
                self._log.warning("ECG CODE-15% não carregado (pesos ausentes?).")
        except Exception as e:
            self._log.warning("ECG CODE-15% indisponível: %s", e)

        # Derma — EfficientNet-B4 HAM10000
        try:
            from models.derma_model import load_derma
            self.registry["derma"] = load_derma()
            self._log.info("Derma EfficientNet-B4 carregado.")
        except Exception as e:
            self._log.warning("Derma indisponível: %s", e)

        # Pill Identifier — Claude Vision
        try:
            from models.pill_model import load_pill
            self.registry["pill"] = load_pill()
            self._log.info("Pill Identifier (Claude Vision) carregado.")
        except Exception as e:
            self._log.warning("Pill Identifier indisponível: %s", e)

        # Sepse XGBoost — pre-load para evitar cold start nas primeiras chamadas
        try:
            _load_sepse_xgb()
            self.registry["sepse"] = "xgboost-blend-v1.1.0"
            self._log.info("Sepse XGBoost+qSOFA carregado.")
        except Exception as e:
            self._log.warning("Sepse indisponível: %s", e)

        # Tabular — Drive → Volume
        self._load_pkl("cardiac",       "cardiac_xgboost_v2_combined.pkl", "Cardiac XGBoost v2")
        self._load_onnx("risco_materno", "risco_materno_gbm.onnx",         "Risco Materno GBM")
        self._load_xgb_native("mortality", "mortality_xgb_v2.json", "mortality_xgb_v2_colunas.json", "Mortality XGBoost v2", 0.8961, 264)
        self._load_onnx("readmissao",   "readmissao_gbm.onnx",              "Readmissão GBM")
        self._load_onnx("deterioracao", "deterioracao_gbm.onnx",            "Deterioração GBM")
        self._load_onnx("vitaldb",      "vitaldb_ihi_v2.onnx",              "VitalDB IHI v2")
        self._load_onnx("eeg",          "eeg_epilepsy_combined_gbm.onnx",   "EEG Epilepsia GBM")
        self._load_onnx("circor",       "circor_cardiac_gbm.onnx",          "CirCor Cardiac GBM")
        self._load_onnx("lung_sound",   "lung_sound_ensemble.onnx",         "Lung Sound Ensemble")

        # Imagem — PTH EfficientNet-B0 (GPU A10G)
        self._load_pth("onnx_skin",     "skin_efficientnet_b0_gpu.pth",    "Skin EfficientNet-B0")
        self._load_pth("onnx_retina",   "eyepacs_efficientnet_b0.pth",     "EyePACS EfficientNet-B0")
        self._load_pth("onnx_chest",    "chest_xray_efficientnet_b0.pth",  "Chest XR EfficientNet-B0")
        self._load_brain_v2()
        if "onnx_brain_v2" not in self.registry:
            self._load_pth("onnx_brain", "brain_tumor_efficientnet_b0.pth", "Brain Tumor EfficientNet-B0")
        self._load_pth("onnx_fracture", "fractura_efficientnet_b0.pth",    "Fratura EfficientNet-B0")
        self._load_pth("onnx_glaucoma", "glaucoma_efficientnet_b0.pth",    "Glaucoma EfficientNet-B0")
        self._load_pth("mamografia_v5",  "mamografia_v5_best.pth",          "Mamografia V5 EfficientNet-B4 TTA-10")
        self._load_pth("chestxray14",   "chestxray14_densenet121_v2.pth",  "ChestX-ray14 DenseNet-121")
        self._load_onnx("onnx_tc_v4",    "tc_cranio_v4_ct.onnx",    "TC Crânio v4 CT")
        self._load_onnx("onnx_mammo_v2", "mamografia_v2_busi.onnx", "Mamografia v2 BUSI")

    # ── FastAPI ASGI App ────────────────────────────────────────────────────

    @modal.asgi_app(label="api")
    def serve(self):
        from fastapi import FastAPI, File, HTTPException, Request, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel

        fast_app = FastAPI(
            title="Aldora AI Models — Modal GPU",
            version="4.0.0",
            description="31 modelos IA médica. GPU A10G. CFM 2.454/2026.",
        )
        fast_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )

        # ── Schemas ─────────────────────────────────────────────────────────

        class ImageRequest(BaseModel):
            image_base64: str
            image_mime: str = "image/jpeg"
            paciente_id: str | None = None

        class ECGRequest(BaseModel):
            fc: float; pr_ms: float; qrs_ms: float; qtc_ms: float
            rr_variability: float = 0.0; paciente_id: str | None = None

        class ECGSignalsRequest(BaseModel):
            leads_100hz: list[list[float]]
            fc: float | None = 0.0; pr_ms: float | None = 0.0
            qrs_ms: float | None = 0.0; qtc_ms: float | None = 0.0
            rr_variability: float | None = 0.0; paciente_id: str | None = None

        class SepseRequest(BaseModel):
            hr: float | None = None; o2sat: float | None = None
            temp: float | None = None; sbp: float | None = None
            map: float | None = None; dbp: float | None = None
            resp: float | None = None; bun: float | None = None
            glucose: float | None = None; lactate: float | None = None
            potassium: float | None = None; creatinine: float | None = None
            hct: float | None = None; hgb: float | None = None
            wbc: float | None = None; platelets: float | None = None
            age: float | None = None; gender: float | None = None
            icu_los_days: float | None = None; paciente_id: str | None = None

        class CardiacRequest(BaseModel):
            murmur_presence: float | None = 0.0; age_years: float | None = 30.0
            sex: float | None = 0.0; pregnancy_status: float | None = 0.0
            campaign: float | None = 1.0

        class RiscoMaternoRequest(BaseModel):
            age: float | None = None
            systolic_bp: float | None = None
            diastolic_bp: float | None = None
            blood_sugar_mgdl: float | None = None
            body_temp_celsius: float | None = None
            heart_rate: float | None = None

        class MortalityRequest(BaseModel):
            valores: dict[str, float | None] = {}

        class ReadmissaoRequest(BaseModel):
            age: float | None = 50.0; length_of_stay: float | None = 5.0
            charlson_index: float | None = 0.0; lace_score: float | None = 5.0
            previous_admissions: float | None = 0.0; diagnosis_icd: float | None = 0.0
            acuity: float | None = 1.0

        class DeterioracaoRequest(BaseModel):
            news2_score: float | None = 0.0; delta_fc: float | None = 0.0
            delta_fr: float | None = 0.0; delta_pas: float | None = 0.0
            delta_temperatura: float | None = 0.0; delta_saturacao: float | None = 0.0
            delta_glasgow: float | None = 0.0; suporte_o2: float | None = 0.0

        class VitalDBRequest(BaseModel):
            mbp: float | None = 80.0; hr: float | None = 70.0
            spo2: float | None = 98.0; etco2: float | None = 35.0
            bis: float | None = 50.0; temperature: float | None = 36.5
            age: float | None = 50.0; asa_class: float | None = 2.0
            operation_type: float | None = 0.0

        class EEGRequest(BaseModel):
            delta_power: float | None = 0.0; theta_power: float | None = 0.0
            alpha_power: float | None = 0.0; beta_power: float | None = 0.0
            gamma_power: float | None = 0.0; spectral_entropy: float | None = 0.0
            hjorth_mobility: float | None = 0.0; hjorth_complexity: float | None = 0.0

        class CircorRequest(BaseModel):
            age_months: float | None = 120.0; weight: float | None = 20.0
            height: float | None = 110.0; sex: float | None = 0.0
            murmur_locations: float | None = 0.0
            systolic_murmur_timing: float | None = 0.0
            diastolic_murmur_timing: float | None = 0.0

        class LungSoundRequest(BaseModel):
            mfcc_mean: float | None = 0.0; mfcc_std: float | None = 0.0
            spectral_centroid: float | None = 0.0
            zero_crossing_rate: float | None = 0.0; rms_energy: float | None = 0.0

        reg = self.registry

        # ── Endpoints ────────────────────────────────────────────────────────

        @fast_app.get("/health")
        def health():
            return {
                "status": "ok",
                "modelos": {k: "carregado" for k in reg},
                "total": len(reg),
                "versao": "4.0.0-modal-gpu",
                "gpu": "A10G",
                "timestamp": _ts(),
            }

        # ── Legados ──────────────────────────────────────────────────────────

        @fast_app.post("/v1/chexpert")
        async def chexpert(req: ImageRequest):
            m = reg.get("chexpert")
            if m is None:
                raise HTTPException(503, "CheXpert não carregado.")
            resultado = m.predict(base64.b64decode(req.image_base64))
            top_conf = resultado[0]["probabilidade"] if resultado else 0.0
            return {"resultado": resultado, "confianca": round(top_conf, 4),
                    "modelo": "torchxrayvision/densenet121-res224-all",
                    "aviso": DISCLAIMER_CFM, "timestamp": _ts()}

        @fast_app.post("/v1/ecg")
        async def ecg_rule(req: ECGRequest):
            m = reg.get("ecg")
            if m is None:
                raise HTTPException(503, "ECG não carregado.")
            resultado = m.predict(req.fc, req.pr_ms, req.qrs_ms, req.qtc_ms, req.rr_variability)
            achados = resultado.get("achados", [])
            return {"resultado": resultado,
                    "confianca": round(achados[0]["probabilidade"], 4) if achados else 0.0,
                    "modelo": "rule-based-ecg-v1", "aviso": DISCLAIMER_CFM, "timestamp": _ts()}

        @fast_app.post("/v1/ecg-code")
        async def ecg_code(req: ECGSignalsRequest):
            """Endpoint para o modelo CODE-15% (Brasileiro, ResNet1D).
            Requer sinais brutos de 12 derivações. Se receber 100Hz, faz upsampling para 500Hz."""
            m = reg.get("ecg_code15")
            if m is None:
                raise HTTPException(503, "ECG CODE-15% não carregado. Verifique se os pesos foram baixados do Drive.")

            import numpy as np
            from scipy import signal as scipy_signal

            signals = np.array(req.leads_100hz, dtype=np.float32)

            # Validação básica de shape
            if signals.shape[0] != 12:
                raise HTTPException(400, f"Esperado 12 derivações, recebido {signals.shape[0]}.")

            # Upsampling de 100Hz para 500Hz se necessário (heurística: se duração for ~10s a 100Hz = 1000 pontos)
            # O CODE-15% espera 5000 pontos (10s @ 500Hz).
            if signals.shape[1] < 2000:  # Provavelmente 100Hz ou menos
                # Upsample por fator de 5 (100 -> 500)
                signals = scipy_signal.resample(signals, 5000, axis=1)
            elif signals.shape[1] != 5000:
                # Se não for exatamente 5000, resample para 5000
                signals = scipy_signal.resample(signals, 5000, axis=1)

            resultado = m.predict(signals)
            resultado["timestamp"] = _ts()
            return {"resultado": resultado, "aviso": DISCLAIMER_CFM}

        @fast_app.post("/ecg/analyze-signals")
        async def ecg_signals(req: ECGSignalsRequest):
            m = reg.get("ecg")
            if m is None:
                raise HTTPException(503, "ECG não carregado.")
            result = m.predict_from_signals(
                leads_100hz=req.leads_100hz,
                fc=req.fc or 0.0, pr_ms=req.pr_ms or 0.0,
                qrs_ms=req.qrs_ms or 0.0, qtc_ms=req.qtc_ms or 0.0,
                rr_variability=req.rr_variability or 0.0,
            )
            return {"resultado": result,
                    "confianca": round(float(result.get("top_probabilidade", 0.0)), 4),
                    "modelo": result.get("modelo", "lgbm_ptbxl"),
                    "aviso": DISCLAIMER_CFM, "timestamp": _ts()}

        @fast_app.post("/v1/derma")
        async def derma(req: ImageRequest):
            m = reg.get("derma")
            if m is None:
                raise HTTPException(503, "Derma não carregado.")
            resultado = m.predict(base64.b64decode(req.image_base64))
            top_conf = resultado[0]["probabilidade"] if resultado else 0.0
            return {"resultado": resultado, "confianca": round(top_conf, 4),
                    "modelo": "efficientnet_b4-ham10000", "aviso": DISCLAIMER_CFM, "timestamp": _ts()}

        @fast_app.post("/v1/pill")
        async def pill(req: ImageRequest):
            m = reg.get("pill")
            if m is None:
                raise HTTPException(503, "Pill Identifier não carregado. Verifique ANTHROPIC_API_KEY.")
            resultado = m.predict(base64.b64decode(req.image_base64), req.image_mime)
            nivel = resultado.get("confianca", "baixa")
            return {"resultado": resultado,
                    "confianca": {"alta": 0.90, "media": 0.65, "baixa": 0.35}.get(nivel, 0.35),
                    "modelo": "claude-sonnet-4-6-vision", "aviso": DISCLAIMER_CFM, "timestamp": _ts()}

        # ── Sepse ─────────────────────────────────────────────────────────────

        @fast_app.post("/v1/sepse")
        async def sepse_v1(req: SepseRequest):
            dados_en = {
                "HR": req.hr, "O2Sat": req.o2sat, "Temp": req.temp,
                "SBP": req.sbp, "MAP": req.map, "DBP": req.dbp, "Resp": req.resp,
                "BUN": req.bun, "Glucose": req.glucose, "Lactate": req.lactate,
                "Potassium": req.potassium, "Creatinine": req.creatinine,
                "Hct": req.hct, "Hgb": req.hgb, "WBC": req.wbc,
                "Platelets": req.platelets, "Age": req.age, "Gender": req.gender,
                "ICULOS": req.icu_los_days,
                "GCS": None,
            }
            resultado = predict_sepsis_hybrid(dados_en)
            return {"resultado": resultado, "confianca": resultado["probabilidade"],
                    "modelo": "xgboost-blend-qsofa-v1.1.0",
                    "aviso": DISCLAIMER_CFM, "timestamp": _ts()}

        @fast_app.post("/sepsis/predict")
        async def sepsis_predict(request: Request):
            try:
                dados = await request.json()
                dados_en = {_SEPSE_PT_MAP.get(k, k): v for k, v in dados.items()}
                resultado = predict_sepsis_hybrid(dados_en)
                return {"success": True, "data": resultado}
            except Exception as e:
                return {"success": False, "error": str(e), "score": 50, "risco": "indeterminado"}

        # ── Tabular ─────────────────────────────────────────────────────────

        @fast_app.post("/cardiac/predict")
        async def cardiac(req: CardiacRequest):
            m = reg.get("cardiac")
            if m is None:
                raise HTTPException(503, "Cardiac XGBoost não carregado.")
            features = ["murmur_presence", "age_years", "sex", "pregnancy_status", "campaign"]
            values = [req.murmur_presence, req.age_years, req.sex, req.pregnancy_status, req.campaign]
            return _tabular_infer(m, values, features, "cardiac_xgboost_v2_combined.pkl")

        # Retreinado na Sessão 67; AUC 0.9265; GradientBoostingClassifier
        # n_estimators=300 max_depth=5 learning_rate=0.05 random_state=42;
        # 452 linhas após deduplicação dos datasets Maternal Health Risk;
        # alvo binário RiskLevel==2 (risco materno ALTO); NÃO prediz
        # pré-eclâmpsia — falta proteinúria e idade gestacional no dataset;
        # StandardScaler embutido no grafo ONNX; notebook e dossiê em
        # ALDORA HEALTH/dossiê_treinamento/; substitui preeclampsia_gbm.onnx,
        # que decidia por glicemia e errava hipertensão grave.
        @fast_app.post("/risco-materno/predict")
        async def risco_materno(req: RiscoMaternoRequest):
            m = reg.get("risco_materno")
            if m is None:
                raise HTTPException(503, "Risco Materno GBM não carregado.")
            missing = [
                name for name, val in [
                    ("age", req.age),
                    ("systolic_bp", req.systolic_bp),
                    ("diastolic_bp", req.diastolic_bp),
                    ("blood_sugar_mgdl", req.blood_sugar_mgdl),
                    ("body_temp_celsius", req.body_temp_celsius),
                    ("heart_rate", req.heart_rate),
                ] if val is None
            ]
            if missing:
                raise HTTPException(400, f"Campos ausentes: {', '.join(missing)}")
            body_temp_f = req.body_temp_celsius * 9.0 / 5.0 + 32.0
            bs_mmol = req.blood_sugar_mgdl / 18.0
            features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
            values = [req.age, req.systolic_bp, req.diastolic_bp, bs_mmol, body_temp_f, req.heart_rate]
            return _tabular_infer(m, values, features, "risco_materno_gbm.onnx")

        # Retreinado na Sessão 67 com a versão COMPLETA do WiDS Datathon 2020
        # (186 colunas) — o modelo anterior usava a versão reduzida de 85 colunas,
        # SEM nenhuma laboratorial; AUC 0.8961 contra 0.8886; XGBoost em GPU,
        # 2,6 segundos de treino; 106 colunas (features + indicadores __medido
        # de ausencia); booster CORTADO em 264 árvores (best_iteration+1) —
        # sem o corte, predict usaria 304 e divergiria 0,069 em silêncio;
        # aceita NaN nativamente, NÃO imputar; recall 0.74 no óbito com
        # precisão 0.34, o scale_pos_weight deslocou para sensibilidade;
        # notebook e dossiê em ALDORA HEALTH/dossiê_treinamento/.
        @fast_app.post("/mortality/predict")
        async def mortality(req: MortalityRequest):
            import numpy as np
            import xgboost as xgb

            m = reg.get("mortality")
            if m is None:
                raise HTTPException(503, "Mortality XGBoost v2 não carregado.")

            if not req.valores:
                raise HTTPException(400, "Dicionário 'valores' vazio — fornecer ao menos uma feature.")

            colunas = m["colunas"]
            desconhecidas = [k for k in req.valores if k not in colunas]
            if desconhecidas:
                raise HTTPException(400, f"Chaves desconhecidas: {', '.join(desconhecidas)}")

            valores_ordenados = []
            features_ausentes = []
            features_utilizadas = 0

            for col in colunas:
                val = req.valores.get(col)
                if val is None:
                    valores_ordenados.append(float('nan'))
                    features_ausentes.append(col)
                else:
                    valores_ordenados.append(float(val))
                    features_utilizadas += 1

            arr = np.array([valores_ordenados], dtype=np.float32)
            probabilidade = float(m["booster"].predict(xgb.DMatrix(arr))[0])
            score = int(probabilidade * 100)

            # Faixas de triagem baseadas na mortalidade de 8,63% do dataset
            if probabilidade < 0.10:
                risco = "baixo"
                cor = "verde"
                mensagem = "Risco baixo de mortalidade."
            elif probabilidade < 0.25:
                risco = "moderado"
                cor = "amarelo"
                mensagem = "Risco moderado — monitorar evolução."
            elif probabilidade < 0.50:
                risco = "alto"
                cor = "laranja"
                mensagem = "Risco alto — avaliação clínica urgente recomendada."
            else:
                risco = "critico"
                cor = "vermelho"
                mensagem = "Risco crítico — intervenção imediata considerada."

            return {
                "success": True,
                "data": {
                    "score": score,
                    "probabilidade": probabilidade,
                    "risco": risco,
                    "cor": cor,
                    "mensagem": mensagem,
                    "features_utilizadas": features_utilizadas,
                    "features_ausentes": features_ausentes,
                    "auc_modelo": 0.8961,
                    "versao_modelo": "mortality_xgb_v2.json",
                    "dataset_treino": "WiDS Datathon 2020 (MIT GOSSIS)",
                    "disclaimer": "Conforme CFM 2.454/2026: esta é uma ferramenta de apoio à decisão clínica e não substitui avaliação médica presencial."
                }
            }

        @fast_app.post("/readmissao/predict")
        async def readmissao(req: ReadmissaoRequest):
            m = reg.get("readmissao")
            if m is None:
                raise HTTPException(503, "Readmissão GBM não carregado.")
            features = ["age", "length_of_stay", "charlson_index", "lace_score",
                        "previous_admissions", "diagnosis_icd", "acuity"]
            values = [req.age, req.length_of_stay, req.charlson_index, req.lace_score,
                      req.previous_admissions, req.diagnosis_icd, req.acuity]
            return _tabular_infer(m, values, features, "readmissao_gbm.onnx")

        @fast_app.post("/deterioracao/predict")
        async def deterioracao(req: DeterioracaoRequest):
            m = reg.get("deterioracao")
            if m is None:
                raise HTTPException(503, "Deterioração GBM não carregado.")
            features = ["news2_score", "delta_fc", "delta_fr", "delta_pas",
                        "delta_temperatura", "delta_saturacao", "delta_glasgow", "suporte_o2"]
            values = [req.news2_score, req.delta_fc, req.delta_fr, req.delta_pas,
                      req.delta_temperatura, req.delta_saturacao, req.delta_glasgow, req.suporte_o2]
            return _tabular_infer(m, values, features, "deterioracao_gbm.onnx")

        @fast_app.post("/vitaldb/predict")
        async def vitaldb(req: VitalDBRequest):
            m = reg.get("vitaldb")
            if m is None:
                raise HTTPException(503, "VitalDB não carregado.")
            features = ["mbp", "hr", "spo2", "etco2", "bis", "temperature",
                        "age", "asa_class", "operation_type"]
            values = [req.mbp, req.hr, req.spo2, req.etco2, req.bis, req.temperature,
                      req.age, req.asa_class, req.operation_type]
            return _tabular_infer(m, values, features, "vitaldb_ihi_v2.onnx")

        @fast_app.post("/eeg/predict")
        async def eeg(req: EEGRequest):
            m = reg.get("eeg")
            if m is None:
                raise HTTPException(503, "EEG não carregado.")
            features = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
                        "spectral_entropy", "hjorth_mobility", "hjorth_complexity"]
            values = [req.delta_power, req.theta_power, req.alpha_power, req.beta_power, req.gamma_power,
                      req.spectral_entropy, req.hjorth_mobility, req.hjorth_complexity]
            return _tabular_infer(m, values, features, "eeg_epilepsy_combined_gbm.onnx")

        @fast_app.post("/circor/predict")
        async def circor(req: CircorRequest):
            m = reg.get("circor")
            if m is None:
                raise HTTPException(503, "CirCor não carregado.")
            features = ["age_months", "weight", "height", "sex", "murmur_locations",
                        "systolic_murmur_timing", "diastolic_murmur_timing"]
            values = [req.age_months, req.weight, req.height, req.sex, req.murmur_locations,
                      req.systolic_murmur_timing, req.diastolic_murmur_timing]
            return _tabular_infer(m, values, features, "circor_cardiac_gbm.onnx")

        @fast_app.post("/lung/predict")
        async def lung(req: LungSoundRequest):
            m = reg.get("lung_sound")
            if m is None:
                raise HTTPException(503, "Lung Sound não carregado.")
            features = ["mfcc_mean", "mfcc_std", "spectral_centroid", "zero_crossing_rate", "rms_energy"]
            values = [req.mfcc_mean, req.mfcc_std, req.spectral_centroid, req.zero_crossing_rate, req.rms_energy]
            return _tabular_infer(m, values, features, "lung_sound_ensemble.onnx")

        # ── Imagem — multipart/form-data ─────────────────────────────────────

        @fast_app.post("/image/chest-xray")
        async def img_chest(image: UploadFile = File(...)):
            s = reg.get("onnx_chest")
            if s is None:
                raise HTTPException(503, "Chest XR não carregado.")
            return _onnx_infer(s, await image.read(), _ONNX_LABELS["chest_xray"], "chest_xray_efficientnet_b0.pth")

        @fast_app.post("/image/skin")
        async def img_skin(image: UploadFile = File(...)):
            s = reg.get("onnx_skin")
            if s is None:
                raise HTTPException(503, "Skin não carregado.")
            return _onnx_infer(s, await image.read(), _ONNX_LABELS["skin"], "skin_efficientnet_b0_gpu.pth")

        @fast_app.post("/image/retinopathy")
        async def img_retina(image: UploadFile = File(...)):
            s = reg.get("onnx_retina")
            if s is None:
                raise HTTPException(503, "Retinopatia não carregado.")
            return _onnx_infer(s, await image.read(), _ONNX_LABELS["retinopathy"], "eyepacs_efficientnet_b0.pth")

        @fast_app.post("/image/brain-tumor")
        async def img_brain(image: UploadFile = File(...)):
            # Preferir v2 (ONNX binário sigmoid, AUC=0.9995) se disponível
            s_v2 = reg.get("onnx_brain_v2")
            if s_v2 is not None:
                img_bytes = await image.read()
                inp = _preprocess_onnx(img_bytes)
                out = s_v2.run(None, {s_v2.get_inputs()[0].name: inp})[0][0]
                raw = float(out) if out.size == 1 else float(out[0])
                prob = float(1.0 / (1.0 + np.exp(-raw)))
                label = "anormal" if prob > 0.5 else "normal"
                conf = round(prob if label == "anormal" else 1.0 - prob, 4)
                return {"predicao": label, "confianca": conf,
                        "scores": {"normal": round(1.0 - prob, 4), "anormal": round(prob, 4)},
                        "modelo_versao": "brain_tumor_v2_combined.onnx",
                        "disclaimer": DISCLAIMER_SHORT}
            # Fallback: PTH 4-classes (50% acurácia — aguardando .data file do v2)
            s = reg.get("onnx_brain")
            if s is None:
                raise HTTPException(503, "Brain Tumor não carregado.")
            return _onnx_infer(s, await image.read(), _ONNX_LABELS["brain_tumor"], "brain_tumor_efficientnet_b0.pth")

        @fast_app.post("/image/fracture")
        async def img_fracture(image: UploadFile = File(...)):
            s = reg.get("onnx_fracture")
            if s is None:
                raise HTTPException(503, "Fratura não carregado.")
            return _onnx_infer(s, await image.read(), _ONNX_LABELS["fracture"], "fractura_efficientnet_b0.pth")

        @fast_app.post("/image/glaucoma")
        async def img_glaucoma(image: UploadFile = File(...)):
            s = reg.get("onnx_glaucoma")
            if s is None:
                raise HTTPException(503, "Glaucoma não carregado.")
            return _onnx_infer(s, await image.read(), _ONNX_LABELS["glaucoma"], "glaucoma_efficientnet_b0.pth")

        @fast_app.post("/image/mamografia-v5")
        async def img_mammo_v5(image: UploadFile = File(...)):
            s = reg.get("mamografia_v5")
            if s is None:
                raise HTTPException(503, "Mamografia V5 não carregado.")
            return _mamografia_v5_tta_infer(s, await image.read())

        @fast_app.post("/image/tc-cranio-v4")
        async def img_tc_v4(image: UploadFile = File(...)):
            s = reg.get("onnx_tc_v4")
            if s is None:
                raise HTTPException(503, "TC Crânio v4 não carregado.")
            img_bytes = await image.read()
            inp = _preprocess_onnx(img_bytes)
            out = s.run(None, {s.get_inputs()[0].name: inp})[0][0]
            raw = float(out) if out.size == 1 else float(out[0])
            prob = float(1.0 / (1.0 + np.exp(-raw)))
            label = "anormal" if prob > 0.5 else "normal"
            conf = round(prob if label == "anormal" else 1.0 - prob, 4)
            return {"predicao": label, "confianca": conf,
                    "scores": {"normal": round(1.0 - prob, 4), "anormal": round(prob, 4)},
                    "modelo_versao": "tc_cranio_v4_ct.onnx", "auc_treino": 0.9732,
                    "disclaimer": DISCLAIMER_SHORT}

        @fast_app.post("/image/mamografia-v2")
        async def img_mammo_v2(image: UploadFile = File(...)):
            """Ultrassom mamário BUSI — NÃO é mamografia convencional."""
            s = reg.get("onnx_mammo_v2")
            if s is None:
                raise HTTPException(503, "Mamografia v2 não carregado.")
            img_bytes = await image.read()
            inp = _preprocess_onnx(img_bytes)
            out = s.run(None, {s.get_inputs()[0].name: inp})[0][0]
            raw = float(out) if out.size == 1 else float(out[0])
            prob = float(1.0 / (1.0 + np.exp(-raw)))
            label = "anormal" if prob > 0.5 else "normal"
            conf = round(prob if label == "anormal" else 1.0 - prob, 4)
            return {"predicao": label, "confianca": conf,
                    "scores": {"normal": round(1.0 - prob, 4), "anormal": round(prob, 4)},
                    "modelo_versao": "mamografia_v2_busi.onnx", "auc_treino": 0.9388,
                    "modalidade": "ultrassom_mamario",
                    "disclaimer": DISCLAIMER_SHORT}

        return fast_app
