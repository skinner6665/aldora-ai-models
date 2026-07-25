"""
CheXpert — torchxrayvision DenseNet densenet121-res224-all
14 patologias pulmonares. AUC médio ~0.84. Treinado em múltiplos datasets.
"""
import io
import numpy as np
from typing import Any


LABELS_PT: dict[str, str] = {
    "Atelectasis": "Atelectasia",
    "Cardiomegaly": "Cardiomegalia",
    "Consolidation": "Consolidação",
    "Edema": "Edema Pulmonar",
    "Effusion": "Derrame Pleural",
    "Emphysema": "Enfisema",
    "Fibrosis": "Fibrose Pulmonar",
    "Hernia": "Hérnia",
    "Infiltration": "Infiltrado Pulmonar",
    "Mass": "Massa Pulmonar",
    "Nodule": "Nódulo Pulmonar",
    "Pleural_Thickening": "Espessamento Pleural",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumotórax",
    # Labels extras (CheXpert original)
    "Enlarged_Cardiomediastinum": "Alarg. Cardiomediastinal",
    "Fracture": "Fratura de Costela",
    "Lung_Lesion": "Lesão Pulmonar",
    "Lung_Opacity": "Opacidade Pulmonar",
    "No_Finding": "Sem Achado Relevante",
    "Pleural_Effusion": "Derrame Pleural",
    "Pleural_Other": "Outro Achado Pleural",
    "Support_Devices": "Dispositivos de Suporte",
}


class ChexpertModel:
    def __init__(self, model: Any):
        self._model = model
        self._pathologies: list[str] = list(model.pathologies)

    def predict(self, img_bytes: bytes) -> list[dict]:
        import torch
        import torchxrayvision as xrv
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img_np = np.array(img, dtype=np.float32)

        # Normaliza para [-1024, 1024] conforme torchxrayvision
        img_np = xrv.datasets.normalize(img_np, 255)
        img_np = img_np[None, ...]  # (1, H, W)

        # Crop/resize via transforms do torchxrayvision
        img_np = xrv.datasets.XRayCenterCrop()(img_np)
        img_np = xrv.datasets.XRayResizer(224)(img_np)

        img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1, 1, 224, 224)

        with torch.no_grad():
            logits = self._model(img_t)
            probs = torch.sigmoid(logits).squeeze().tolist()

        if not isinstance(probs, list):
            probs = [probs]

        results = []
        for label, prob in zip(self._pathologies, probs):
            if isinstance(prob, (float, int)):
                results.append({
                    "patologia": LABELS_PT.get(label, label),
                    "patologia_en": label,
                    "probabilidade": round(float(prob), 4),
                    "positivo": float(prob) > 0.5,
                })

        return sorted(results, key=lambda x: -x["probabilidade"])


def load_chexpert() -> ChexpertModel:
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return ChexpertModel(model)
