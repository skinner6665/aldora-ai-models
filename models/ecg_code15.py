"""
ECG CODE-15% Fase B18 — Modelo Brasileiro (UFMG / Telehealth Network)

Arquitetura: ResNet1D treinada no CODE-15% completo (339.939 exames, 231.488 pacientes).
Classes (ordem posicional fixa): 1dAVb, RBBB, LBBB, SB, ST, AF.
Entrada: 12 derivações, janela de 2934 amostras, taxa nativa 400 Hz.
Normalização: robusta por derivação com vetores FIXOS da fase B.
Ativação: sigmoid independente por alvo (multi-rótulo).

Conformidade: CFM 2.454/2026 — apoio à decisão, nunca diagnóstico autônomo.
Fonte: Ribeiro et al., CODE-15%, Zenodo 10.5281/zenodo.4916206 (CC BY 4.0).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_MODEL_DIR = Path(__file__).parent
_WEIGHTS_DIR_CANDIDATES = [
    Path("/models"),
    Path("/app/models"),
    _MODEL_DIR,
]

# ── Classes do modelo (ordem posicional fixa do treino fase B) ────────────────
CODE15_CLASSES = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
CODE15_CLASSES_PT = {
    "1dAVb": "Bloqueio atrioventricular de 1º grau",
    "RBBB":  "Bloqueio de ramo direito",
    "LBBB":  "Bloqueio de ramo esquerdo",
    "SB":    "Bradicardia sinusal",
    "ST":    "Taquicardia sinusal",
    "AF":    "Fibrilação atrial",
}

# ── Normalização robusta FIXA da fase B ───────────────────────────────────────
# Estes valores são da FASE B. Pesos code15_faseB18_fold*.pt só funcionam
# com eles. Usar normalização da fase A degrada em silêncio.
_NORM_MED = np.array([
    -0.041656494140625, -0.06475830078125, -0.01953125,
     0.0648193359375,   -0.0059051513671875, -0.03631591796875,
     0.024688720703125, -0.0197601318359375, -0.03790283203125,
    -0.07421875,        -0.085693359375,    -0.07818603515625,
], dtype=np.float32).reshape(12, 1)

_NORM_IQR = np.array([
    0.3060302734375, 0.371826171875, 0.35479736328125,
    0.28521728515625, 0.3006591796875, 0.32452392578125,
    0.360107421875, 0.4505615234375, 0.505615234375,
    0.4815673828125, 0.4278564453125, 0.3836669921875,
], dtype=np.float32).reshape(12, 1)

_SR_NATIVO = 400
_JANELA = 2934
_NUM_CLASSES = 6


class ECGCode15Model(nn.Module):
    """
    Arquitetura FIEL à treinada em ahs74_ecg_code15_faseB18.ipynb, célula 10
    (classe `ResNet1D` no notebook). Apesar do nome, NÃO tem conexão residual
    — é puramente sequencial: stem -> 4 blocos (conv-bn-relu-dropout-conv-bn-
    -relu) -> head. Os nomes de atributo (stem, b, head) são exigidos pelo
    load_state_dict(strict=True) contra os pesos salvos — NÃO renomear nem
    "simplificar" para nomenclatura torchvision-style. Isso já causou uma
    incompatibilidade de arquitetura silenciosa (AHS-77): a versão anterior
    desta classe usava conv1/bn1/layer1-4/fc com shortcut residual, uma
    arquitetura diferente da que gerou os pesos.
    Input: (batch, 12, 2934) — 12 leads, janela de 2934 amostras a 400 Hz.
    Output: (batch, 6) logits.
    """
    def __init__(self, num_classes: int = _NUM_CLASSES, ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, ch, kernel_size=17, stride=2, padding=8),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
        )

        def _bloco(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(i, o, kernel_size=17, stride=2, padding=8),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(o, o, kernel_size=17, stride=1, padding=8),
                nn.BatchNorm1d(o),
                nn.ReLU(),
            )

        self.b = nn.Sequential(
            _bloco(ch, 128),
            _bloco(128, 196),
            _bloco(196, 256),
            _bloco(256, 320),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(320, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.b(self.stem(x)))


class ECGCode15Predictor:
    """
    Gerenciador de inferência para o modelo CODE-15% fase B.
    Ensemble de 5 folds. Falha explícita se qualquer peso estiver ausente.
    """

    # Filenames esperados na ordem dos folds
    _EXPECTED_WEIGHTS = [
        "code15_faseB18_fold0.pt",
        "code15_faseB18_fold1.pt",
        "code15_faseB18_fold2.pt",
        "code15_faseB18_fold3.pt",
        "code15_faseB18_fold4.pt",
    ]

    def __init__(self) -> None:
        self.models: list[ECGCode15Model] = []
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.loaded = False
        self._load_models()

    def _find_weights_dir(self) -> Path | None:
        for base_dir in _WEIGHTS_DIR_CANDIDATES:
            if not base_dir.exists():
                continue
            # Verifica se pelo menos o fold0 existe neste diretório
            if (base_dir / self._EXPECTED_WEIGHTS[0]).exists():
                return base_dir
        return None

    def _load_models(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("[ECG-CODE15] PyTorch não disponível.")

        weights_dir = self._find_weights_dir()
        if weights_dir is None:
            raise FileNotFoundError(
                f"[ECG-CODE15] Pesos ausentes. Esperado: {self._EXPECTED_WEIGHTS[0]} "
                f"em um de {_WEIGHTS_DIR_CANDIDATES}. O endpoint /v1/ecg-code15 deve "
                "retornar 503 até que os pesos sejam montados."
            )

        missing = []
        for fname in self._EXPECTED_WEIGHTS:
            if not (weights_dir / fname).exists():
                missing.append(fname)
        if missing:
            raise FileNotFoundError(
                f"[ECG-CODE15] Pesos ausentes em {weights_dir}: {missing}. "
                "Todos os 5 folds são obrigatórios. Sem qualquer fold, o ensemble "
                "está incompleto e o endpoint deve retornar 503."
            )

        print(f"[ECG-CODE15] Carregando 5 folds de: {weights_dir}")

        for fname in self._EXPECTED_WEIGHTS:
            wf = weights_dir / fname
            model = ECGCode15Model(num_classes=_NUM_CLASSES)
            state_dict = torch.load(wf, map_location="cpu", weights_only=True)

            # Handle DataParallel wrapper
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Remove prefix 'module.' se presente
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v

            # Gate de cardinalidade: camada final DEVE ter 6 saídas.
            # strict=True (padrão) já garante isso; load_state_dict falha
            # se houver mismatch. NUNCA usar strict=False.
            model.load_state_dict(new_state_dict)
            # Linear final vive em model.head[2] (Sequential: [0]=AvgPool,
            # [1]=Flatten, [2]=Linear) — não existe mais model.fc nesta
            # arquitetura. Ver docstring de ECGCode15Model.
            assert model.head[2].out_features == _NUM_CLASSES, (
                f"[ECG-CODE15] Gate de cardinalidade falhou: head[2].out_features="
                f"{model.head[2].out_features}, esperado {_NUM_CLASSES}. "
                "Pesos incompatíveis com o contrato."
            )

            model.to(self.device)
            model.eval()
            self.models.append(model)
            print(f"[ECG-CODE15] Fold carregado: {fname}")

        self.loaded = True
        print(f"[ECG-CODE15] Pronto. {len(self.models)} folds no ensemble. Device: {self.device}")

    def preprocess(self, signals: np.ndarray) -> torch.Tensor:
        """
        Pré-processamento fase B.
        Input: numpy array (12, 2934) ou (batch, 12, 2934).
        Normalização robusta por derivação com vetores FIXOS.
        """
        if signals.ndim == 2:
            signals = np.expand_dims(signals, axis=0)

        # Normalização robusta: (x - mediana) / IQR, vetores fixos da fase B
        normalized = (signals.astype(np.float32) - _NORM_MED) / _NORM_IQR
        return torch.tensor(normalized, dtype=torch.float32).to(self.device)

    @staticmethod
    def _adjust_window(signals: np.ndarray) -> np.ndarray:
        """
        Ajusta sinal para a janela de 2934 amostras.
        Mais curto: padding centrado simétrico com zeros.
        Mais longo: recorte CENTRAL GEOMÉTRICO — (n - 2934) // 2.

        NÃO é recorte por máscara de validade: nenhuma máscara é calculada
        aqui. Coincide com o centro válido enquanto o padding for simétrico
        (caso do CODE-15%, 95,9% dos exames em 581+2934+581=4096). Com
        padding assimétrico, pode incluir amostras de padding nas bordas.
        """
        n_samples = signals.shape[-1]
        if n_samples == _JANELA:
            return signals
        if n_samples < _JANELA:
            pad_total = _JANELA - n_samples
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(signals, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)
        # Mais longo: recorte central
        start = (n_samples - _JANELA) // 2
        return signals[..., start:start + _JANELA]

    def predict(self, signals: list[list[float]] | np.ndarray, sr: float | None = None) -> dict:
        """
        Inferência com ensemble (média das probabilidades sigmoid).
        Input: lista de 12 listas ou numpy array (12, N).
        sr: taxa de amostragem do sinal. Se fornecida e != 400, rejeita com erro.
        """
        if not self.loaded:
            return {"erro": "Modelo CODE-15% não carregado.", "modelo": "code15_unavailable"}

        if isinstance(signals, list):
            signals_np = np.array(signals, dtype=np.float32)
        else:
            signals_np = signals.astype(np.float32)

        # Validação de shape básico
        if signals_np.ndim != 2 or signals_np.shape[0] != 12:
            return {
                "erro": f"Shape inválido: esperado (12, N), recebido {signals_np.shape}",
                "modelo": "code15_shape_error",
            }

        # Gate de taxa de amostragem
        if sr is not None and abs(sr - _SR_NATIVO) > 1.0:
            return {
                "erro": (
                    f"Taxa de amostragem {sr} Hz não suportada. "
                    f"O modelo foi treinado a {_SR_NATIVO} Hz nativo. "
                    "Reamostrar ANTES de enviar ou usar sinal a 400 Hz."
                ),
                "modelo": "code15_sr_error",
            }

        # Ajuste de janela
        signals_np = self._adjust_window(signals_np)

        # Validação pós-ajuste
        if signals_np.shape != (12, _JANELA):
            return {
                "erro": f"Shape após ajuste: {signals_np.shape}, esperado (12, {_JANELA})",
                "modelo": "code15_shape_error",
            }

        input_tensor = self.preprocess(signals_np)

        all_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(input_tensor)
                # Sigmoid INDEPENDENTE por alvo (multi-rótulo). Nunca softmax.
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        # Média do ensemble
        avg_probs = np.mean(all_probs, axis=0)[0]

        # Montar resultado — apenas probabilidades, sem limiar nem alerta automático.
        # Ordem POSICIONAL FIXA de CODE15_CLASSES, conforme contrato. NÃO ranquear:
        # consumidor que indexe por posição depende dela, e ordenar por
        # probabilidade insinua um "achado principal" que não foi calibrado.
        resultados = []
        for i, cls in enumerate(CODE15_CLASSES):
            resultados.append({
                "classe": cls,
                "classe_pt": CODE15_CLASSES_PT[cls],
                "probabilidade": round(float(avg_probs[i]), 4),
            })

        return {
            "modelo": "code15_resnet1d_ensemble_faseB18",
            "dataset": "CODE-15% (UFMG/Telehealth)",
            "sr_nativo": _SR_NATIVO,
            "janela": _JANELA,
            "resultados": resultados,
            "disclaimer": (
                "Resultado gerado por IA (CODE-15%) como apoio à decisão clínica. "
                "Não substitui avaliação médica. CFM 2.454/2026."
            ),
        }


def load_ecg_code15() -> ECGCode15Predictor:
    return ECGCode15Predictor()