"""HAM10000 — EfficientNet-B4, 7 classes de lesões cutâneas."""
import io
from typing import Any
import numpy as np

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

CLASSES_PT = {
    "MEL": "Melanoma",
    "NV": "Nevo Melanocítico (pinta benigna)",
    "BCC": "Carcinoma Basocelular",
    "AKIEC": "Queratose Actínica / Carcinoma in situ",
    "BKL": "Queratose Benigna (seborreica/solar)",
    "DF": "Dermatofibroma",
    "VASC": "Lesão Vascular (angioma, granuloma)",
}

# Condutas sugeridas por classe (nunca comunicar diretamente ao paciente)
CONDUTA = {
    "MEL": "Encaminhar para dermatologista com urgência — suspeita de melanoma.",
    "NV": "Acompanhamento dermatológico de rotina.",
    "BCC": "Encaminhar para dermatologista — biópsia recomendada.",
    "AKIEC": "Encaminhar para dermatologista — lesão pré-neoplásica.",
    "BKL": "Acompanhamento clínico; biópsia se atípica.",
    "DF": "Benigno; excisão apenas se sintomático.",
    "VASC": "Avaliação dermatológica para confirmação.",
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

        return [
            {
                "classe": CLASSES_PT[c],
                "classe_en": c,
                "probabilidade": round(p, 4),
                "conduta_medica": CONDUTA[c],
            }
            for c, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1])
        ]


def load_derma() -> DermaModel:
    import torch, timm
    import torchvision.transforms as T
    import os

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=7)

    weights_path = os.environ.get("DERMA_WEIGHTS_PATH", MODEL_WEIGHTS_PATH)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)

    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return DermaModel(model, transform)
