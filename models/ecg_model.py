"""
ECG — Análise rule-based clínica (sem modelo pesado).
Parâmetros de entrada: FC, PR (ms), QRS (ms), QTc (ms), variabilidade RR (ms).
Limites baseados em critérios ACC/AHA 2019 + SBC Diretrizes ECG 2023.
"""


class ECGModel:
    """
    Análise clínica de ECG por regras determinísticas.
    Não requer GPU nem download de modelo.
    """

    def predict(
        self,
        fc: float,
        pr_ms: float,
        qrs_ms: float,
        qtc_ms: float,
        rr_variability: float,
    ) -> dict:
        achados: list[dict] = []
        alertas: list[str] = []

        # ── Frequência Cardíaca ───────────────────────────────────────────────
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
        else:
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

        # ── Intervalo PR (normal: 120–200 ms) ────────────────────────────────
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
        if pr_ms > 0.0 and pr_ms < 120.0:
            achados.append({
                "achado": "PR Curto — Considerar Síndrome de Pré-excitação (WPW)",
                "achado_en": "short_pr_wpw",
                "probabilidade": 0.60,
                "positivo": True,
            })
            alertas.append("PR < 120 ms — excluir Síndrome de Wolff-Parkinson-White")

        # ── Duração QRS (normal: < 120 ms) ───────────────────────────────────
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

        # ── QTc — Bazett (normal: ≤ 440 ms H / ≤ 460 ms F) ─────────────────
        if qtc_ms > 500.0:
            achados.append({
                "achado": "QTc Muito Prolongado — Risco Alto de Torsades de Pointes",
                "achado_en": "prolonged_qt_critical",
                "probabilidade": 0.97,
                "positivo": True,
            })
            alertas.append(
                "QTc > 500 ms — ALTO RISCO de Torsades de Pointes: revisar TODOS os fármacos "
                "que prolongam QT, corrigir hipocalemia/hipomagnesemia"
            )
        elif qtc_ms > 460.0:
            achados.append({
                "achado": "QTc Prolongado — Monitorar Risco de Torsades",
                "achado_en": "prolonged_qt",
                "probabilidade": 0.82,
                "positivo": True,
            })
            alertas.append(
                "QTc > 460 ms — Monitorar; revisar fármacos que prolongam QT (quinolonas, "
                "antifúngicos azólicos, antipsicóticos, amiodarona)"
            )
        elif qtc_ms > 0.0 and qtc_ms < 350.0:
            achados.append({
                "achado": "QTc Curto — Considerar Síndrome do QT Curto",
                "achado_en": "short_qt",
                "probabilidade": 0.55,
                "positivo": True,
            })
            alertas.append("QTc < 350 ms — excluir Síndrome do QT Curto (risco de FV)")

        # ── Variabilidade RR ──────────────────────────────────────────────────
        # Variabilidade RR > 120 ms com FC elevada → suspeita de FA/flutter
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
                    "considerar Holter 24h se ECG não confirmar"
                )
            else:
                achados.append({
                    "achado": "Irregularidade RR — Monitorar (arritmia benigna vs. FA lenta)",
                    "achado_en": "rr_irregularity",
                    "probabilidade": 0.55,
                    "positivo": True,
                })

        # Ordena por probabilidade decrescente
        achados.sort(key=lambda x: -x["probabilidade"])

        return {
            "achados": achados,
            "alertas": alertas,
            "parametros": {
                "fc_bpm": fc,
                "pr_ms": pr_ms,
                "qrs_ms": qrs_ms,
                "qtc_ms": qtc_ms,
                "rr_variabilidade_ms": rr_variability,
            },
        }


def load_ecg() -> ECGModel:
    return ECGModel()
