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


class ResNetBlock1D(nn.Module):
    """Bloco residual 1D para ECG."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=17, stride=stride, padding=8, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=17, stride=1, padding=8, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ECGCode15Model(nn.Module):
    """
    ResNet adaptada para CODE-15%.
    Input: (batch, 12, 2934) — 12 leads, janela de 2934 amostras a 400 Hz.
    Output: (batch, 6) logits.
    """
    def __init__(self, num_classes: int = _NUM_CLASSES):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
            assert model.fc.out_features == _NUM_CLASSES, (
                f"[ECG-CODE15] Gate de cardinalidade falhou: fc.out_features="
                f"{model.fc.out_features}, esperado {_NUM_CLASSES}. "
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