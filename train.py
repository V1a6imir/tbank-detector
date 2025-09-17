import torch
from ultralytics import YOLO
import os


_orig_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = unsafe_load



def main():
    exp_name = "train" 

    # Загружаем предобученные веса из Ultralytics (маленькая модель)
    model = YOLO("yolov8n.pt")  

    results = model.train(
        data="data/yolo/data1.yaml",  
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name=exp_name,
        device=0  # GPU, если доступно
    )


    best_weights = os.path.join("runs/detect", exp_name, "weights", "best.pt")
    print(f"✅ Лучшие веса: {best_weights}")


    model = YOLO(best_weights)

    print("➡️ Экспорт модели в ONNX...")
    model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)


if __name__ == "__main__":
    main()
