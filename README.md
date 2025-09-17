# 🏦 TBank Logo Detector

Решение задачи детекции логотипа T-Банка с использованием **YOLOv8**.  
Модель обучена на вручную и автоматически размеченных данных (zero-shot OWLv2 + CLIP), дополненных аугментациями и ребалансировкой выборки.

---

## 📌 Структура репозитория

```
.
├── app/                      # REST API сервис (FastAPI + Uvicorn)
│   ├── inference.py          # Класс TBankDetector для инференса
│   ├── main.py               # API сервер
│   └── schemas.py            # Pydantic-схемы
├── data/
│   └── validation/           # Валидационный датасет (images, labels, validation.yaml)
├── docker/
│   └── Dockerfile            # Docker-сборка
├── eval/
│   └── evaluate.py           # Скрипт для оценки IoU, Precision, Recall, F1
├── models/                   # Папка для весов (перед запуском скачать!)
├── scripts/
│   ├── prepare_zeroshot_labels.py  # Авторазметка OWLv2+CLIP
│   └── validate.py           # Запуск валидации YOLO
├── requirements.txt          # Зависимости для API
├── requirements.train.txt    # Зависимости для обучения
├── train.py                  # Скрипт для обучения модели
└── README.md                 # Документация
```



---

## 📊 Подход к решению

1. **Сбор данных**  
   - Использовались сырые изображения логотипа T-Банк и негативные примеры.  
   - Применялась **zero-shot разметка** (OWLv2 + CLIP) для автоматического сбора части обучающей выборки.  
   - Добавлены **негативные примеры** (Tinkoff, пустые сцены), чтобы модель могла отличать.  

2. **Аугментации**  
   - Использованы Albumentations: flips, rotation, affine, color jitter, blur, brightness/contrast.  

3. **Ребалансировка**  
   - Ограничено количество негативных примеров в train/val.  
   - Обеспечено равномерное распределение размеченных изображений.  

4. **Обучение**  
   - Базовая модель: `yolov8n.pt`.  
   - Тренировка: 100 эпох, `imgsz=640`, датасет — `data/yolo/data1.yaml`.  
   - Экспорт лучших весов в **ONNX** и **PyTorch (.pt)**.  

5. **Метрики на валидации**

| Метрика         | Значение |
|----------------|----------|
| Precision      | 0.924    |
| Recall         | 0.786    |
| F1@0.5         | 0.849    |
| mAP@0.5        | 0.867    |
| mAP@0.5:0.95   | 0.577    |

---

## 🚀 Запуск API

### 1. Скачать веса модели

Скачайте обученные веса и поместите их в папку `models/`:

- [best.pt (PyTorch)](https://drive.google.com/file/d/1SOwO4YIYLH1hZsfEtvb5G49BMsfU0aVh/view?usp=drive_link)  
- [best.onnx (ONNX)](https://drive.google.com/file/d/1iEsbGdPO2zp_7HKO2nV5yNwuEE8ZAJmI/view?usp=drive_link)  

Структура:
```
models/
├── best.pt
└── best.onnx

```


---

### 2. Сборка Docker-образа
```
git clone https://github.com/V1a6imir/tbank-detector.git
cd tbank-detector
docker build -t tbank-detector -f docker/Dockerfile .
```

### 3. Запуск контейнера

```
docker run --rm -p 8000:8000 tbank-detector

```
После запуска сервис будет доступен по адресу:

👉 http://localhost:8000/docs

Можно протестировать эндпоинт /detect, загрузив изображение.

### 📈 Валидация модели

Для запуска скрипта валидации на отложенной выборке, завершите процесс ctrl+C а затем оставясь в папке проекта введите:

```
docker run --rm -v "$(pwd)/results:/app/results" tbank-detector python3 scripts/validate.py
```
Результаты:

Метрики сохраняются в results/metrics.json.

Визуализации предсказаний — в runs/detect/val_with_preds/.

### 📚 Используемые технологии
YOLOv8 (Ultralytics)

 - FastAPI

- ONNX Runtime

- Albumentations

- OWLv2 + CLIP

### 🔮 Возможные улучшения
- Добавить hard negative mining (поиск сложных отрицательных примеров).

- Использовать более тяжёлую модель (yolov8m / yolov8l) для повышения точности.

- Применить distillation для облегчённой модели и ускорения API.

- Настроить CI/CD пайплайн для автоматического деплоя.

### 📜 Лицензия
MIT — свободное использование в учебных и исследовательских целях.

