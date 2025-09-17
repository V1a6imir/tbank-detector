# 🏦 TBank Logo Detector

Решение задачи детекции логотипа T-Банка с использованием **YOLOv8**.  
Модель обучена на вручную и автоматически размеченных данных (zero-shot OWLv2 + CLIP), дополненных аугментациями и ребалансировкой выборки.  

## 📌 Структура репозитория
.
├── app/ # REST API сервис (FastAPI + Uvicorn)
│ ├── inference.py # Класс TBankDetector для инференса
│ ├── main.py # API сервер
│ └── schemas.py # Pydantic-схемы
├── data/
│ └── validation/ # Валидационный датасет (images, labels, validation.yaml)
├── docker/
│ └── Dockerfile # Docker-сборка
├── eval/
│ └── evaluate.py # Скрипт для оценки IoU, Precision, Recall, F1
├── models/ # Папка для весов (перед запуском скачать!)
├── scripts/
│ ├── prepare_zeroshot_labels.py # Авторазметка OWLv2+CLIP
│ └── validate.py # Запуск валидации YOLO
├── requirements.txt # зависимости для API
├── requirements.train.txt# зависимости для обучения
├── train.py # скрипт для обучения модели
└── README.md # документация

markdown
Копировать код

---

## 📊 Подход к решению

1. **Сбор данных**  
   - Использовались сырые изображения логотипа T-Банк и негативные примеры.  
   - Дополнительно применялась **zero-shot разметка** (OWLv2 + CLIP), чтобы автоматически собрать часть обучающей выборки.  
   - Отдельно собраны **негативные примеры** (Tinkoff, пустые сцены), чтобы модель училась отличать.  

2. **Аугментации**  
   Использованы Albumentations: flips, rotation, affine, color jitter, blur, brightness/contrast.  

3. **Ребалансировка**  
   - Ограничили количество негативных примеров в train/val.  
   - Добавили равномерное распределение размеченных изображений.  

4. **Обучение**  
   - Базовая модель: `yolov8n.pt`.  
   - Тренировка на датасете (`data/yolo/data1.yaml`) 100 эпох, `imgsz=640`.  
   - Лучшая модель экспортирована в **ONNX** и **PyTorch (.pt)**.  

5. **Метрики на валидации**  
Precision: 0.924
Recall: 0.786
F1@0.5: 0.849
mAP@0.5: 0.867
mAP@0.5:0.95 0.577

yaml
Копировать код

---

## 🚀 Запуск API

### 1. Скачать веса модели
Скачайте обученные веса и поместите их в папку `models/`:

- [best.pt (PyTorch)](https://files.catbox.moe/XXXXXXXX.pt)  
- [best.onnx (ONNX)](https://files.catbox.moe/YYYYYYYY.onnx)  

Итоговая структура:
models/
├── best.pt
└── best.onnx

perl
Копировать код

### 2. Сборка Docker-образа

docker build -t tbank-detector -f docker/Dockerfile .
3. Запуск контейнера
bash
Копировать код
docker run --rm -p 8000:8000 tbank-detector
После этого сервис будет доступен по адресу:
👉 http://localhost:8000/docs

Там можно протестировать эндпоинт /detect, загрузив изображение.

📈 Валидация модели
Для запуска скрипта валидации на отложенной выборке:

bash
Копировать код
docker run --rm \
    -v "$(pwd)/results:/app/results" \
    tbank-detector \
    python3 scripts/validate.py
Результаты:

Метрики сохраняются в results/metrics.json.

Визуализации предсказаний — в runs/detect/val_with_preds/.

📚 Используемые технологии
YOLOv8 (Ultralytics)

FastAPI

ONNX Runtime

Albumentations

OWLv2 + CLIP

🔮 Возможные улучшения
Добавить hard negative mining (поиск сложных отрицательных примеров).

Использовать более тяжёлую модель (yolov8m / yolov8l) для прироста точности.

Попробовать distillation на более лёгкую модель для ускорения API.

Добавить CI/CD пайплайн для автоматического деплоя.

📜 Лицензия
MIT — свободное использование в учебных и исследовательских целях.
