# Hugging Face Spaces + Render 通用 Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用 Docker 缓存
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码（代码变动不会 invalidate 依赖缓存）
COPY backend/ .

# HF Spaces 用 $PORT，Render 也用 $PORT，本地默认 8000
CMD python main.py
