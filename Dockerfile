# ── Mac M3: forzar plataforma AMD64 para TensorFlow ───────────────────────────
FROM --platform=linux/amd64 python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    tensorflow-cpu==2.16.2 \
    numpy \
    pydantic==2.7.4 \
    pillow \
    python-multipart

# Copiar proyecto
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]