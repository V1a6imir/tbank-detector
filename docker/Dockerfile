FROM python:3.11


WORKDIR /app

# Системные зависимости для OpenCV и Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 curl wget libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и модели
COPY app ./app
COPY models ./models
COPY scripts ./scripts
COPY data ./data
COPY README.md ./

# Добавляем в PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8000

# Запуск API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "10"]
