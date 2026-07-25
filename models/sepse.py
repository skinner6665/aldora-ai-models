"""XGBoost Sepse — PhysioNet Challenge 2019 (60k pacientes UTI)."""
import os, numpy as np
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
    if prob < 0.2:
        return "baixo"
    elif prob < 0.4:
        return "moderado"
    elif prob < 0.7:
        return "alto"
    return "critico"


class SepseModel:
    def __init__(self, booster: Any):
        self._booster = booster
        try:
            import shap
            self._explainer = shap.TreeExplainer(booster)
        except Exception:
            self._explainer = None

    def predict(self, features: list[float]) -> dict:
        import xgboost as xgb
        import numpy as np

        X = np.array(features, dtype=np.float32).reshape(1, -1)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        prob = float(self._booster.predict(dmat)[0])

        shap_top5 = []
        if self._explainer is not None:
            try:
                shap_vals = self._explainer.shap_values(X)[0]
                top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
                shap_top5 = [
                    {"variavel": FEATURE_NAMES[i], "impacto": round(float(shap_vals[i]), 4)}
                    for i in top_idx
                ]
            except Exception:
                pass

        return {
            "prob": round(prob, 4),
            "score": round(prob * 100, 1),
            "risco": _prob_to_risco(prob),
            "shap_top5": shap_top5,
        }


def load_sepse() -> SepseModel:
    import xgboost as xgb

    booster = xgb.Booster()
    weights_path = os.environ.get("SEPSE_MODEL_PATH", MODEL_PATH)
    if os.path.exists(weights_path):
        booster.load_model(weights_path)
    else:
        # Modelo demo com parâmetros padrão (sem treinamento real)
        # Produção: executar train_sepse.py com dados PhysioNet 2019
        booster = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="auc",
        )
        booster.fit(
            np.zeros((2, len(FEATURE_NAMES))),
            [0, 1],
        )
        booster = booster.get_booster()

    return SepseModel(booster)
