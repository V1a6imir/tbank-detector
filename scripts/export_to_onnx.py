import torch
from ultralytics import YOLO

# Глобальный патч: всегда грузим с weights_only=False
_orig_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = unsafe_load

# Загружаем модель
model = YOLO("/Users/vladimirmonastirsky/Downloads/tbank-detector/runs/detect/train/weights/best.pt")

# Экспортируем в ONNX
model.export(format="onnx", imgsz=640, opset=12)
