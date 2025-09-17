from pydantic import BaseModel, Field
from typing import List, Optional


class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0, example=10)
    y_min: int = Field(..., description="Верхняя координата", ge=0, example=20)
    x_max: int = Field(..., description="Правая координата", ge=0, example=110)
    y_max: int = Field(..., description="Нижняя координата", ge=0, example=120)


class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")


class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(
        default_factory=list,
        description="Список найденных логотипов",
        example=[{"bbox": {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 120}}]
    )


class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки", example="Invalid file format")
    detail: Optional[str] = Field(None, description="Дополнительная информация", example="Only JPEG/PNG supported")
