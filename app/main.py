from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import DetectionResponse, Detection, BoundingBox, ErrorResponse
from app.inference import TBankDetector
import numpy as np
import cv2
import os
from pathlib import Path
import time

# --- Инициализация приложения ---
app = FastAPI(title="T-Bank Logo Detector")

# Универсальный путь к весам
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "models" / "best.onnx"

detector = TBankDetector(weights_path=str(WEIGHTS_PATH))

# --- Поддерживаемые форматы ---
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-Банка на изображении
    """

    if file.content_type not in SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Unsupported file format",
                detail=f"Allowed formats: {', '.join(SUPPORTED_FORMATS)}"
            ).dict()
        )

    try:
        start_time = time.time()

        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        # Запуск детекции
        boxes = detector.detect(img, conf_threshold=0.25)

        detections = [
            Detection(bbox=BoundingBox(
                x_min=int(x1), y_min=int(y1), x_max=int(x2), y_max=int(y2)
            )) for (x1, y1, x2, y2, conf) in boxes
        ]

        elapsed = time.time() - start_time
        if elapsed > 10:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="Processing time exceeded",
                    detail=f"Detection took {elapsed:.2f} seconds (>10s limit)"
                ).dict()
            )

        return DetectionResponse(detections=detections)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Detection failed",
                detail=str(e)
            ).dict()
        )
