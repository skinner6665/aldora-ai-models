"""
HAM10000 — EfficientNet-B4 (timm) pretrained.
7 classes de lesões cutâneas. Backbone ImageNet → fine-tuning HAM10000.
Condutas médicas: não comunicar diagnóstico diretamente ao paciente (CFM 2.454/2026).
"""
import io
import os
from typing import Any

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

CLASSES_PT: dict[str, str] = {
    "MEL": "Melanoma",
    "NV": "Nevo Melanocítico (pinta benigna)",
    "BCC": "Carcinoma Basocelular",
    "AKIEC": "Queratose Actínica / Carcinoma in situ",
    "BKL": "Queratose Benigna (seborreica / solar)",
    "DF": "Dermatofibroma",
    "VASC": "Lesão Vascular (angioma, granuloma piogênico)",
}

CONDUTA: dict[str, str] = {
    "MEL": "Encaminhar ao dermatologista com urgência — suspeita de melanoma maligno.",
    "NV": "Acompanhamento dermatológico de rotina; biópsia se sinais ABCDE alterados.",
    "BCC": "Encaminhar ao dermatologista — biópsia e tratamento cirúrgico recomendados.",
    "AKIEC": "Encaminhar ao dermatologista — lesão pré-neoplásica; crioterapia ou excisão.",
    "BKL": "Acompanhamento clínico; biópsia se atipias ou mudança rápida.",
    "DF": "Lesão benigna; excisão cirúrgica apenas se sintomático ou cosmético.",
    "VASC": "Avaliação dermatológica para confirmação; pode necessitar de laser ou excisão.",
}

MALIGNIDADE: dict[str, str] = {
    "MEL": "alta",
    "BCC": "moderada",
    "AKIEC": "moderada",
    "NV": "baixa",
    "BKL": "baixa",
    "DF": "baixa",
    "VASC": "baixa",
}

MODEL_WEIGHTS_PATH = "checkpoints/derma_efficientnet_b4.pth"


class DermaModel:
    def __init__(self, model: Any, transform: Any):
        self._model = model
        self._transform = transform

    def predict(self, img_bytes: bytes) -> list[dict]:
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        t = self._transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(t)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()

        if not isinstance(probs, list):
            probs = [probs]

        results = [
            {
                "classe": CLASSES_PT[c],
                "classe_en": c,
                "probabilidade": round(float(p), 4),
                "malignidade": MALIGNIDADE[c],
                "conduta_medica": CONDUTA[c],
            }
            for c, p in zip(CLASSES, probs)
        ]
        return sorted(results, key=lambda x: -x["probabilidade"])


def load_derma() -> DermaModel:
    import torch
    import timm
    import torchvision.transforms as T

    # Backbone ImageNet pretrained; head adaptado para 7 classes
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=7)

    # Carrega pesos fine-tuned; falha explícita se ausente
    weights_path = os.environ.get("DERMA_WEIGHTS_PATH", MODEL_WEIGHTS_PATH)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"pesos do derma ausentes em {weights_path}. O modelo NAO esta "
            "treinado: a cabeca de 7 classes seria aleatoria e o endpoint "
            "devolveria melanoma e carcinoma a partir de ruido. Medido na "
            "AHS-71: delta 0,2011 entre conteineres. Enquanto o .pth nao "
            "existir, /v1/derma deve retornar 503."
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    model.eval()

    transform = T.Compose([
        T.Resize((380, 380)),
        T.CenterCrop(380),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return DermaModel(model, transform)
