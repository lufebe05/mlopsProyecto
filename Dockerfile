FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente, artefactos del modelo y plantillas
COPY src/ ./src/
COPY models/ ./models/
COPY templates/ ./templates/
COPY app_docker.py .

EXPOSE 8000

CMD ["python", "app_docker.py"]