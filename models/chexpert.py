"""CheXpert — torchxrayvision DenseNet121, 14 patologias pulmonares."""
import io, numpy as np
from typing import Any

LABELS_CHEXPERT = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion", "Lung_Opacity",
    "No_Finding", "Pleural_Effusion", "Pleural_Other", "Pneumonia",
    "Pneumothorax", "Support_Devices",
]

LABELS_PT = {
    "Atelectasis": "Atelectasia",
    "Cardiomegaly": "Cardiomegalia",
    "Consolidation": "Consolidação",
    "Edema": "Edema Pulmonar",
    "Enlarged_Cardiomediastinum": "Alarg. Cardiomediastinal",
    "Fracture": "Fratura de Costela",
    "Lung_Lesion": "Lesão Pulmonar",
    "Lung_Opacity": "Opacidade Pulmonar",
    "No_Finding": "Sem Achado Relevante",
    "Pleural_Effusion": "Derrame Pleural",
    "Pleural_Other": "Outro Achado Pleural",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumotórax",
    "Support_Devices": "Dispositivos de Suporte",
}


class ChexpertModel:
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
            probs = torch.sigmoid(logits).squeeze().tolist()

        results = []
        for label, prob in zip(LABELS_CHEXPERT, probs):
            if isinstance(prob, float):
                results.append({
                    "patologia": LABELS_PT.get(label, label),
                    "patologia_en": label,
                    "probabilidade": round(prob, 4),
                    "positivo": prob > 0.5,
                })
        return sorted(results, key=lambda x: -x["probabilidade"])


def load_chexpert() -> ChexpertModel:
    import torchxrayvision as xrv
    import torchvision.transforms as T

    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    return ChexpertModel(model, transform)
