"""ECG-FM — bowang-lab/ecg-fm, HuggingFace."""
from typing import Any
import numpy as np


ACHADOS_PT = {
    "normal_sinus_rhythm": "Ritmo Sinusal Normal",
    "sinus_tachycardia": "Taquicardia Sinusal",
    "sinus_bradycardia": "Bradicardia Sinusal",
    "atrial_fibrillation": "Fibrilação Atrial",
    "atrial_flutter": "Flutter Atrial",
    "ventricular_tachycardia": "Taquicardia Ventricular",
    "ventricular_fibrillation": "Fibrilação Ventricular",
    "left_bundle_branch_block": "BRE (Bloqueio Ramo Esquerdo)",
    "right_bundle_branch_block": "BRD (Bloqueio Ramo Direito)",
    "first_degree_av_block": "BAV 1º Grau",
    "st_elevation": "Elevação ST",
    "st_depression": "Infradesnivelamento ST",
    "t_wave_inversion": "Inversão Onda T",
    "left_ventricular_hypertrophy": "HVE (Hipertrofia Ventricular Esquerda)",
    "right_ventricular_hypertrophy": "HVD (Hipertrofia Ventricular Direita)",
    "prolonged_qt": "QT Prolongado",
    "paced_rhythm": "Ritmo de Marcapasso",
    "premature_atrial_complex": "Extrassístole Atrial",
    "premature_ventricular_complex": "Extrassístole Ventricular",
}


class ECGModel:
    def __init__(self, pipe: Any):
        self._pipe = pipe

    def predict(self, sinal: list[float], derivacoes: int, freq_hz: int) -> list[dict]:
        import torch

        arr = np.array(sinal, dtype=np.float32)
        if derivacoes > 0:
            arr = arr.reshape(derivacoes, -1)

        resultado = self._pipe(arr)
        achados = []
        for item in resultado:
            label = item.get("label", "")
            score = float(item.get("score", 0.0))
            achados.append({
                "achado": ACHADOS_PT.get(label, label),
                "achado_en": label,
                "probabilidade": round(score, 4),
                "positivo": score > 0.5,
            })
        return sorted(achados, key=lambda x: -x["probabilidade"])


def load_ecg() -> ECGModel:
    from transformers import pipeline as hf_pipeline

    token = __import__("os").environ.get("HUGGINGFACE_TOKEN")
    pipe = hf_pipeline(
        "text-classification",
        model="bowang-lab/ecg-fm",
        token=token,
        top_k=None,
    )
    return ECGModel(pipe)
