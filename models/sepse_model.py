"""
XGBoost Sepse — PhysioNet Challenge 2019 (~60k pacientes UTI).
40 variáveis laboratoriais e sinais vitais.
Fallback para modelo demo se checkpoint não disponível.
"""
import os
import numpy as np
from typing import Any

MODEL_PATH = "checkpoints/sepse_xgb.json"

FEATURE_NAMES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "HoraNoUTI",
]


def _prob_to_risco(prob: float) -> str:
    if prob < 0.20:
        return "baixo"
    elif prob < 0.40:
        return "moderado"
    elif prob < 0.70:
        return "alto"
    return "critico"


class SepseModel:
    def __init__(self, booster: Any, demo_mode: bool = False):
        self._booster = booster
        self._demo_mode = demo_mode
        self._explainer = None
        if not demo_mode:
            try:
                import shap
                self._explainer = shap.TreeExplainer(booster)
            except Exception:
                pass

    def predict(self, features: list) -> dict:
        import xgboost as xgb

        X = np.array(features, dtype=np.float32).reshape(1, -1)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        prob = float(self._booster.predict(dmat)[0])

        shap_top5: list[dict] = []
        if self._explainer is not None:
            try:
                shap_vals = self._explainer.shap_values(X)[0]
                top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
                shap_top5 = [
                    {
                        "variavel": FEATURE_NAMES[int(i)],
                        "impacto": round(float(shap_vals[i]), 4),
                    }
                    for i in top_idx
                ]
            except Exception:
                pass

        return {
            "prob": round(prob, 4),
            "score": round(prob * 100.0, 1),
            "risco": _prob_to_risco(prob),
            "variaveis_impacto": shap_top5,
            "modo_demo": self._demo_mode,
        }


def load_sepse() -> SepseModel:
    import xgboost as xgb

    weights_path = os.environ.get("SEPSE_MODEL_PATH", MODEL_PATH)

    if os.path.exists(weights_path):
        booster = xgb.Booster()
        booster.load_model(weights_path)
        return SepseModel(booster, demo_mode=False)

    # Modelo demo: treinado em 2 pontos fictícios para retornar probabilidade coerente
    # Para produção: executar train_sepse.py com dados PhysioNet 2019
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
    )
    X_demo = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    X_demo[1, :] = 1.0
    clf.fit(X_demo, [0, 1])
    return SepseModel(clf.get_booster(), demo_mode=True)
