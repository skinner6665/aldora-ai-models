"""
AHS — Modelo Híbrido Sepse: XGBoost + qSOFA/SIRS
XGBoost: PhysioNet Challenge 2019 + MIMIC-IV Sintético + eICU Demo (AUC=0.8766)
Critérios clínicos: qSOFA (Sepsis-3), SIRS, shock hemodinâmico, hiperlactatemia
Versão: 1.1.0-aldora-blend
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'sepsis_xgboost.pkl'
_model_data = None


def _load_model():
    global _model_data
    if _model_data is None:
        with open(MODEL_PATH, 'rb') as f:
            _model_data = pickle.load(f)
    return _model_data


FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
    'BUN', 'Glucose', 'Lactate', 'Potassium', 'Creatinine',
    'Hct', 'Hgb', 'WBC', 'Platelets',
    'Age', 'Gender', 'ICULOS'
]

FEATURE_MEDIANS: dict[str, float] = {
    'HR': 83.0, 'O2Sat': 97.0, 'Temp': 36.9, 'SBP': 122.0,
    'MAP': 82.0, 'DBP': 63.0, 'Resp': 18.0, 'BUN': 18.0,
    'Glucose': 128.0, 'Lactate': 1.6, 'Potassium': 4.0,
    'Creatinine': 0.9, 'Hct': 31.6, 'Hgb': 10.7,
    'WBC': 10.7, 'Platelets': 213.0,
    'Age': 62.0, 'Gender': 1.0, 'ICULOS': 24.0,
}


def _clinical_score(dados: dict) -> tuple[float, dict]:
    """
    Computa score clínico híbrido: qSOFA + SIRS + componentes de choque.
    Retorna (probabilidade_clinica [0,1], detalhes).

    qSOFA (Sepsis-3): GCS<15, FR≥22, PAS≤100 → cada critério = 1/3
    SIRS: Temp anormal, FC>90, FR>20, WBC anormal → cada critério = 1/4
    Lactato: >2 mmol/L indica hipoperfusão → componente 0-1
    MAP: <65 mmHg = critério de choque séptico → componente 0-1
    SpO2: <94% = hipoxemia → componente 0-1
    """
    qsofa = 0
    gcs = dados.get('GCS')
    resp = dados.get('Resp')
    sbp = dados.get('SBP')
    hr = dados.get('HR')
    temp = dados.get('Temp')
    wbc = dados.get('WBC')
    lactate = dados.get('Lactate')
    map_val = dados.get('MAP')
    spo2 = dados.get('O2Sat')

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

    # Hiperlactatemia (>2 mmol/L = critério de sepse; >4 = choque séptico)
    lactato_comp = 0.0
    if lactate is not None:
        l = float(lactate)
        if l > 2.0:
            lactato_comp = min(1.0, (l - 2.0) / 4.0)

    # Choque hemodinâmico via MAP (<65 = critério Sepsis-3)
    map_comp = 0.0
    if map_val is not None:
        m = float(map_val)
        if m < 65.0:
            map_comp = min(1.0, (65.0 - m) / 30.0)

    # Hipoxemia (SpO2 < 94%)
    spo2_comp = 0.0
    if spo2 is not None:
        s = float(spo2)
        if s < 94.0:
            spo2_comp = min(1.0, (94.0 - s) / 10.0)

    # Pesos: qSOFA 35%, SIRS 20%, Lactato 20%, MAP-choque 15%, SpO2 10%
    clinical_prob = (
        (qsofa / 3.0) * 0.35 +
        (sirs / 4.0) * 0.20 +
        lactato_comp * 0.20 +
        map_comp * 0.15 +
        spo2_comp * 0.10
    )

    detalhes = {
        'qsofa': qsofa,
        'sirs': sirs,
        'lactato_critico': lactate is not None and float(lactate) > 2.0,
        'map_choque': map_val is not None and float(map_val) < 65.0,
        'hipoxemia': spo2 is not None and float(spo2) < 94.0,
    }
    return clinical_prob, detalhes


def predict_sepsis(dados: dict) -> dict:
    model_data = _load_model()
    model = model_data['model']

    dados = dict(dados)

    # Auto-computar MAP quando SBP e DBP disponíveis mas MAP ausente.
    if dados.get('MAP') is None and dados.get('SBP') is not None and dados.get('DBP') is not None:
        sbp = float(dados['SBP'])
        dbp = float(dados['DBP'])
        dados['MAP'] = round((sbp + 2 * dbp) / 3, 1)

    # ICULOS default=0 quando não em contexto UTI (out-of-distribution mas menos distorcido que mediana=24).
    if dados.get('ICULOS') is None:
        dados['ICULOS'] = 0.0

    X = np.array([[
        float(dados[feat] if dados.get(feat) is not None else FEATURE_MEDIANS[feat])
        for feat in FEATURES
    ]])

    xgb_prob = float(model.predict_proba(X)[0][1])

    # Score clínico baseado em qSOFA (Sepsis-3), SIRS e marcadores de choque.
    # Compensa a baixa sensibilidade do XGBoost sem contexto ICULOS.
    clinical_prob, detalhes_clinicos = _clinical_score(dados)

    # Blend: 40% XGBoost (padrões epidemiológicos) + 60% critérios clínicos validados
    blended_prob = xgb_prob * 0.40 + clinical_prob * 0.60
    score = round(blended_prob * 100, 1)

    if blended_prob < 0.10:
        risco = 'baixo'
        cor = 'green'
        mensagem = 'Baixo risco de sepse'
    elif blended_prob < 0.25:
        risco = 'moderado'
        cor = 'yellow'
        mensagem = 'Risco moderado — monitorar sinais vitais e lactato'
    elif blended_prob < 0.50:
        risco = 'alto'
        cor = 'orange'
        mensagem = 'Alto risco — considerar avaliacao imediata e culturas'
    else:
        risco = 'critico'
        cor = 'red'
        mensagem = 'Risco critico — protocolo de sepse recomendado'

    features_presentes = [f for f in FEATURES if dados.get(f) is not None]
    features_imputadas = [f for f in FEATURES if dados.get(f) is None]

    return {
        'score': score,
        'probabilidade': blended_prob,
        'risco': risco,
        'cor': cor,
        'mensagem': mensagem,
        'features_utilizadas': len(features_presentes),
        'features_imputadas': features_imputadas,
        'componentes': {
            'xgboost_score': round(xgb_prob * 100, 1),
            **detalhes_clinicos,
        },
        'auc_modelo': 0.8766,
        'versao_modelo': '1.1.0-aldora-blend-xgb-qsofa',
        'dataset_treino': 'PhysioNet2019 + MIMIC-IV + eICU',
        'disclaimer': 'Ferramenta de apoio à decisão clínica — CFM 2.454/2026',
    }
