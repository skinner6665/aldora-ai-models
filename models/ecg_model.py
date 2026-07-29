"""
ECG — Análise híbrida: LightGBM (PTB-XL) + rule-based fallback.

Modelo primário: LightGBM treinado em 21.388 ECGs do PTB-XL 1.0.3
  AUC-ROC macro: 0.8536 | Classes: CD, HYP, MI, NORM, STTC
  Features: 360 (10 estatísticas + 20 FFT × 12 derivações a 100Hz)

Fallback: regras determinísticas ACC/AHA 2019 + SBC 2023
  Ativado quando sinais brutos não estão disponíveis.

Conformidade: CFM 2.454/2026 — apoio à decisão, nunca diagnóstico autônomo.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

_MODEL_DIR = Path(__file__).parent
_LGBM_PATH = _MODEL_DIR / "ecg_lgbm_ptbxl.pkl"
_LE_PATH = _MODEL_DIR / "ecg_label_encoder.pkl"

# Labels em português para cada superclasse PTB-XL
_CLASS_PT = {
    "NORM": "Ritmo/Morfologia Normal",
    "MI":   "Infarto do Miocárdio",
    "STTC": "Alteração de ST/T",
    "CD":   "Distúrbio de Condução",
    "HYP":  "Hipertrofia",
}

_CLASS_SEVERITY = {
    "NORM": "normal",
    "MI":   "critico",
    "STTC": "alto",
    "CD":   "moderado",
    "HYP":  "moderado",
}


def _extract_features(leads_100hz: list[list[float]]) -> np.ndarray:
    """
    Extrai 360 features de 12 derivações a 100Hz (1000 amostras).
    leads_100hz: lista de 12 listas, cada uma com 1000 floats.
    """
    from scipy import stats as sp_stats
    from scipy.fft import fft as sp_fft

    N_FFT = 20
    feats: list[float] = []
    for lead in leads_100hz:
        s = np.array(lead, dtype=np.float32)
        feats += [
            float(np.mean(s)),
            float(np.std(s)),
            float(sp_stats.skew(s)),
            float(sp_stats.kurtosis(s)),
            float(np.min(s)),
            float(np.max(s)),
            float(np.percentile(s, 25)),
            float(np.percentile(s, 75)),
            float(np.median(s)),
            float(np.sqrt(np.mean(s ** 2))),
        ]
        fft_mag = np.abs(sp_fft(s))[:N_FFT]
        feats += fft_mag.tolist()
    return np.array(feats, dtype=np.float32).reshape(1, -1)


class ECGModel:
    def __init__(self) -> None:
        self._lgbm: Any | None = None
        self._le: Any | None = None
        self._lgbm_available = False
        self._load_lgbm()

    def _load_lgbm(self) -> None:
        try:
            if _LGBM_PATH.exists() and _LE_PATH.exists():
                with open(_LGBM_PATH, "rb") as f:
                    self._lgbm = pickle.load(f)
                with open(_LE_PATH, "rb") as f:
                    self._le = pickle.load(f)
                self._lgbm_available = True
        except Exception as e:
            print(f"[ECG] LightGBM não carregado: {e} — usando fallback rule-based")

    # ── Modo primário: LightGBM ───────────────────────────────────────────
    def predict_from_signals(
        self,
        leads_100hz: list[list[float]],
        fc: float = 0.0,
        pr_ms: float = 0.0,
        qrs_ms: float = 0.0,
        qtc_ms: float = 0.0,
        rr_variability: float = 0.0,
    ) -> dict:
        """
        Predição primária com LightGBM a partir dos sinais brutos das 12 derivações.
        leads_100hz: lista de 12 listas com 1000 samples cada (100Hz, 10s).
        Parâmetros clínicos opcionais para enriquecer o relatório rule-based complementar.
        """
        if not self._lgbm_available:
            return self.predict(fc, pr_ms, qrs_ms, qtc_ms, rr_variability)

        if len(leads_100hz) != 12 or any(len(lead) != 1000 for lead in leads_100hz):
            return {
                "erro": "leads_100hz deve ter 12 derivações × 1000 amostras",
                "modelo": "erro",
            }

        try:
            X = _extract_features(leads_100hz)
            probs = self._lgbm.predict_proba(X)[0]
            classes = self._le.classes_

            resultados = []
            for cls, prob in zip(classes, probs):
                resultados.append({
                    "classe":        cls,
                    "classe_pt":     _CLASS_PT.get(cls, cls),
                    "probabilidade": round(float(prob), 4),
                    "gravidade":     _CLASS_SEVERITY.get(cls, "moderado"),
                })
            resultados.sort(key=lambda x: -x["probabilidade"])

            top = resultados[0]
            alertas: list[str] = []
            if top["classe"] == "MI" and top["probabilidade"] > 0.5:
                alertas.append(
                    "Alta probabilidade de IAM — correlacionar com clínica e enzimas cardíacas"
                )
            if top["classe"] == "STTC" and top["probabilidade"] > 0.4:
                alertas.append(
                    "Alteração de ST/T detectada — avaliar isquemia, digitálicos, distúrbio eletrolítico"
                )
            if top["classe"] == "CD" and top["probabilidade"] > 0.5:
                alertas.append(
                    "Distúrbio de condução — avaliar bloqueio de ramo ou BAV"
                )

            # Complementar com rule-based se parâmetros clínicos disponíveis
            rule_result: dict = {}
            if fc > 0:
                rule_result = self.predict(fc, pr_ms, qrs_ms, qtc_ms, rr_variability)
                alertas += rule_result.get("alertas", [])

            return {
                "modelo":            "lgbm_ptbxl",
                "auc_treino":        0.8536,
                "dataset":           "PTB-XL 1.0.3 (21.388 ECGs)",
                "resultados":        resultados,
                "top_classe":        top["classe_pt"],
                "top_probabilidade": top["probabilidade"],
                "alertas":           list(dict.fromkeys(alertas)),
                "rule_based":        rule_result.get("achados", []),
                "disclaimer": (
                    "Resultado gerado por IA como apoio à decisão clínica. "
                    "Não substitui avaliação médica. CFM 2.454/2026."
                ),
            }
        except Exception as e:
            return {"erro": str(e), "modelo": "lgbm_erro"}

    # ── Modo fallback: rule-based ─────────────────────────────────────────
    def predict(
        self,
        fc: float,
        pr_ms: float,
        qrs_ms: float,
        qtc_ms: float,
        rr_variability: float,
    ) -> dict:
        """Análise rule-based — fallback quando sinais brutos não disponíveis."""
        achados: list[dict] = []
        alertas: list[str] = []

        if 60.0 <= fc <= 100.0:
            achados.append({
                "achado": "Ritmo Sinusal Normal",
                "achado_en": "normal_sinus_rhythm",
                "probabilidade": 0.90,
                "positivo": True,
            })
        elif fc > 100.0:
            p = min(0.97, 0.70 + (fc - 100.0) / 100.0)
            achados.append({
                "achado": "Taquicardia Sinusal",
                "achado_en": "sinus_tachycardia",
                "probabilidade": round(p, 2),
                "positivo": True,
            })
            if fc > 150.0:
                alertas.append(
                    "FC > 150 bpm — considerar TSV ou flutter atrial com condução 2:1"
                )
            if fc > 180.0:
                alertas.append(
                    "FC > 180 bpm — URGENTE: avaliar Fibrilação/Flutter Atrial ou TV"
                )
        elif fc > 0:
            p = min(0.97, 0.70 + (60.0 - fc) / 60.0)
            achados.append({
                "achado": "Bradicardia Sinusal",
                "achado_en": "sinus_bradycardia",
                "probabilidade": round(p, 2),
                "positivo": True,
            })
            if fc < 40.0:
                alertas.append(
                    "FC < 40 bpm — avaliar BAV de alto grau, hipotireoidismo ou intoxicação digitálica"
                )

        if pr_ms > 200.0:
            p_bav = min(0.97, 0.75 + (pr_ms - 200.0) / 200.0)
            achados.append({
                "achado": "Bloqueio AV de 1º Grau",
                "achado_en": "first_degree_av_block",
                "probabilidade": round(p_bav, 2),
                "positivo": True,
            })
        if pr_ms > 320.0:
            achados.append({
                "achado": "Bloqueio AV Avançado — Suspeita (2º/3º grau)",
                "achado_en": "high_degree_av_block",
                "probabilidade": 0.72,
                "positivo": True,
            })
            alertas.append("PR > 320 ms — avaliar BAV de 2º/3º grau")
        if 0.0 < pr_ms < 120.0:
            achados.append({
                "achado": "PR Curto — Considerar Síndrome de Pré-excitação (WPW)",
                "achado_en": "short_pr_wpw",
                "probabilidade": 0.60,
                "positivo": True,
            })
            alertas.append("PR < 120 ms — excluir Síndrome de Wolff-Parkinson-White")

        if 120.0 <= qrs_ms < 150.0:
            achados.append({
                "achado": "Bloqueio de Ramo Incompleto / BRD",
                "achado_en": "right_bundle_branch_block",
                "probabilidade": 0.68,
                "positivo": True,
            })
        elif qrs_ms >= 150.0:
            achados.append({
                "achado": "Bloqueio de Ramo Esquerdo Completo (BRE)",
                "achado_en": "left_bundle_branch_block",
                "probabilidade": 0.80,
                "positivo": True,
            })
            alertas.append(
                "QRS ≥ 150 ms com morfologia BRE — avaliar indicação de CRT (ressincronização)"
            )

        if qtc_ms > 500.0:
            achados.append({
                "achado": "QTc Muito Prolongado — Risco Alto de Torsades de Pointes",
                "achado_en": "prolonged_qt_critical",
                "probabilidade": 0.97,
                "positivo": True,
            })
            alertas.append(
                "QTc > 500 ms — ALTO RISCO de Torsades de Pointes: revisar fármacos que prolongam QT, "
                "corrigir hipocalemia/hipomagnesemia"
            )
        elif qtc_ms > 460.0:
            achados.append({
                "achado": "QTc Prolongado — Monitorar Risco de Torsades",
                "achado_en": "prolonged_qt",
                "probabilidade": 0.82,
                "positivo": True,
            })
            alertas.append("QTc > 460 ms — Monitorar; revisar fármacos que prolongam QT")
        elif 0.0 < qtc_ms < 350.0:
            achados.append({
                "achado": "QTc Curto — Considerar Síndrome do QT Curto",
                "achado_en": "short_qt",
                "probabilidade": 0.55,
                "positivo": True,
            })
            alertas.append("QTc < 350 ms — excluir Síndrome do QT Curto (risco de FV)")

        if rr_variability > 120.0:
            if fc > 80.0:
                achados.append({
                    "achado": "Arritmia Supraventricular — Suspeita de Fibrilação Atrial",
                    "achado_en": "atrial_fibrillation_suspect",
                    "probabilidade": 0.75,
                    "positivo": True,
                })
                alertas.append(
                    "Alta variabilidade RR + FC > 80 bpm — excluir Fibrilação Atrial; "
                    "considerar Holter 24h"
                )
            else:
                achados.append({
                    "achado": "Irregularidade RR — Monitorar (arritmia benigna vs. FA lenta)",
                    "achado_en": "rr_irregularity",
                    "probabilidade": 0.55,
                    "positivo": True,
                })

        achados.sort(key=lambda x: -x["probabilidade"])
        return {
            "modelo":    "rule_based",
            "achados":   achados,
            "alertas":   alertas,
            "parametros": {
                "fc_bpm":              fc,
                "pr_ms":               pr_ms,
                "qrs_ms":              qrs_ms,
                "qtc_ms":              qtc_ms,
                "rr_variabilidade_ms": rr_variability,
            },
            "disclaimer": (
                "Análise rule-based ACC/AHA 2019 + SBC 2023. "
                "Apoio à decisão. CFM 2.454/2026."
            ),
        }


def load_ecg() -> ECGModel:
    return ECGModel()
