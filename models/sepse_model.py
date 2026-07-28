"""
AHS — Modelo XGBoost Sepse
Treinado com: PhysioNet Challenge 2019 + MIMIC-IV Sintético + eICU Demo
AUC-ROC: 0.8766
Features: 19 variáveis clínicas
Versão: 1.0.0-aldora
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

# Medianas de imputação (PhysioNet Challenge 2019)
FEATURE_MEDIANS: dict[str, float] = {
    'HR': 83.0, 'O2Sat': 97.0, 'Temp': 36.9, 'SBP': 122.0,
    'MAP': 82.0, 'DBP': 63.0, 'Resp': 18.0, 'BUN': 18.0,
    'Glucose': 128.0, 'Lactate': 1.6, 'Potassium': 4.0,
    'Creatinine': 0.9, 'Hct': 31.6, 'Hgb': 10.7,
    'WBC': 10.7, 'Platelets': 213.0,
    'Age': 62.0, 'Gender': 1.0, 'ICULOS': 24.0,
}


def predict_sepsis(dados: dict) -> dict:
    """
    Prediz risco de sepse usando XGBoost treinado.

    Input: dict com valores clínicos (qualquer subconjunto das 19 features)
    Output: dict com score, risco, features_utilizadas, versao_modelo
    """
    model_data = _load_model()
    model = model_data['model']

    X = np.array([[
        float(dados.get(feat, FEATURE_MEDIANS[feat]) or FEATURE_MEDIANS[feat])
        for feat in FEATURES
    ]])

    prob = float(model.predict_proba(X)[0][1])
    score = round(prob * 100, 1)

    # Thresholds calibrados para distribuição do PhysioNet 2019 (base rate 2.2% sepse).
    # XGBoost sem calibracao isotonica retorna probabilidades comprimidas (max ~0.65).
    if prob < 0.15:
        risco = 'baixo'
        cor = 'green'
        mensagem = 'Baixo risco de sepse nas proximas 6 horas'
    elif prob < 0.30:
        risco = 'moderado'
        cor = 'yellow'
        mensagem = 'Risco moderado — monitorar sinais vitais e lactato'
    elif prob < 0.55:
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
        'probabilidade': prob,
        'risco': risco,
        'cor': cor,
        'mensagem': mensagem,
        'features_utilizadas': len(features_presentes),
        'features_imputadas': features_imputadas,
        'auc_modelo': 0.8766,
        'versao_modelo': '1.0.0-aldora-xgboost',
        'dataset_treino': 'PhysioNet2019 + MIMIC-IV + eICU',
        'disclaimer': 'Ferramenta de apoio à decisão clínica — CFM 2.454/2026',
    }
