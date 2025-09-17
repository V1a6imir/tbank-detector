# scripts/validate.py
import os
import json
import torch
from ultralytics import YOLO

# --- Патч для PyTorch 2.6 (weights_only=False) ---
_orig_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = unsafe_load
# -------------------------------------------------

def main():
    # Универсальные пути
    weights_path = os.getenv("MODEL_PATH", "models/best.pt")
    data_yaml = os.getenv("DATA_PATH", "data/validation/validation.yaml")

    print(f"[INFO] Using weights: {weights_path}")
    print(f"[INFO] Using dataset: {data_yaml}")

    # Загружаем модель
    model = YOLO(weights_path)

    # Валидация (с сохранением предсказаний)
    results = model.val(
        data=data_yaml,
        split="val",   # берём вал-датасет
        iou=0.5,
        save=True,     # сохраняем картинки с предсказаниями
        save_json=True,
        imgsz=640,
        batch=16,
        project="runs/detect",  # сохраняем в папку runs/detect
        name="val_with_preds"   # имя эксперимента
    )

    # Метрики
    precision = float(results.results_dict["metrics/precision(B)"])
    recall = float(results.results_dict["metrics/recall(B)"])
    map50 = float(results.results_dict["metrics/mAP50(B)"])
    map5095 = float(results.results_dict["metrics/mAP50-95(B)"])
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)

    print("\n=== Validation Results ===")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"F1@0.5:      {f1:.3f}")
    print(f"mAP@0.5:     {map50:.3f}")
    print(f"mAP@0.5:0.95 {map5095:.3f}")
    print("==========================")

    # Сохраняем метрики
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50-95": map5095
    }
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("[INFO] Metrics saved to results/metrics.json")
    print("[INFO] Predictions saved to runs/detect/val_with_preds")

if __name__ == "__main__":
    main()
