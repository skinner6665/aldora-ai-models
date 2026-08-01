FROM python:3.11-slim
# cache-bust: 2026-08-01

WORKDIR /app

# Dependências do sistema para torch, PIL e OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download modelos em cache no build (evita cold start em produção)
RUN python -c "\
import torchxrayvision as xrv; \
m = xrv.models.DenseNet(weights='densenet121-res224-all'); \
print('CheXpert DenseNet OK')"

RUN python -c "\
import timm; \
m = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1000); \
print('EfficientNet-B4 OK')"

# Copia código da aplicação
COPY . .

RUN mkdir -p checkpoints models

# Download dos modelos GBM/XGBoost/ONNX do Google Drive
# Falhas individuais não bloqueiam o build — endpoints retornam 503
RUN python download_models.py || echo "Aviso: alguns modelos não foram baixados. Endpoints afetados retornarão HTTP 503."

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
