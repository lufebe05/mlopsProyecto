# 🚗 Guía de Ejecución y Comandos - Predicción de Precios de Vehículos

Todos los comandos necesarios para configurar el entorno con `uv`, ejecutar el pipeline de Machine Learning, monitorear con MLflow y desplegar la app en local o con Docker.

---

## 🎯 1. Preparación del Entorno

### 1️⃣ Clonar el Repositorio

```bash
# Clonar el proyecto
git clone https://github.com/lufebe05/mlopsProyecto.git

# Entrar al directorio
cd mlopsProyecto
```

### 2️⃣ Crear y Activar el Entorno Virtual con uv

```bash
# Crear el entorno virtual (usará cualquier Python 3.11.x del sistema)
uv venv

# Activar en Windows (PowerShell)
.venv\Scripts\activate

# (Opcional) Activar en Linux/Mac/WSL
# source .venv/bin/activate
```

### 3️⃣ Instalar Dependencias

```bash
# Sincronizar el entorno con uv
uv sync

# Alternativa directa con archivo de requerimientos
uv pip install -r requirements.txt
```

---

## ⚙️ 2. Pipeline de Machine Learning

> ⚠️ **Importante:** Ejecutar los pasos en este orden exacto. El script de descarga debe ejecutarse primero para crear los datos crudos.

### 0️⃣ Ingesta y Descarga de Datos Crudos

Descarga el dataset base y lo ubica en `data/raw/carData.csv`:

```bash
# Ejecutar descarga de datos
python src/data/descarga_data.py
```

### 1️⃣ Preprocesamiento e Ingeniería de Características

Limpia duplicados, agrupa categorías infrecuentes, calcula la antigüedad y genera `models/preprocessor.joblib`:

```bash
# Ejecutar transformación de datos
python src/features/build_features.py
```

### 2️⃣ Entrenamiento Base y Registro en MLflow

Entrena Ridge Regression y Random Forest, calculando métricas reales en INR y registrándolas en `mlflow.db`:

```bash
# Ejecutar entrenamiento
python src/models/train_model.py
```

### 3️⃣ Optimización Bayesiana de Hiperparámetros (Optuna)

Ejecuta 25 trials con validación cruzada (k=5), registra cada corrida en MLflow y exporta `models/random_forest_tuned.joblib`:

```bash
# Ejecutar afinamiento bayesiano
python src/models/tune_model.py
```

### 4️⃣ Probar la Inferencia Local

Verifica que el modelo y el preprocesador serializados devuelvan estimaciones en rupias indias (INR) a partir de datos de prueba:

```bash
# Ejecutar prueba de predicción
python src/models/predict_model.py
```

---

## 📊 3. Monitoreo de Experimentos con MLflow

### 1️⃣ Iniciar el Servidor Gráfico de MLflow

```bash
# Iniciar MLflow apuntando a la base de datos local SQLite
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 2️⃣ Acceder a la Interfaz

- **Abre en tu navegador:** http://127.0.0.1:5000
- **Para detener:** Presiona `Ctrl + C` en la terminal.

---

## 🌐 4. Despliegue de la Aplicación Web

### Opción A: Ejecución Local Directa

```bash
# Iniciar app local en el puerto 8000
python app.py
```

- **Abre en tu navegador:** http://127.0.0.1:8000
- **Detener servidor:** Presiona `Ctrl + C`.

---

### Opción B: Despliegue con Docker

### 1️⃣ Construir la Imagen

```bash
# Construir imagen estándar
docker build -t car-price-predictor:v1 .

# Construir sin usar cache (si hiciste cambios en código o templates)
docker build --no-cache -t car-price-predictor:v1 .
```

### 2️⃣ Ver Imágenes

```bash
# Ver imágenes en el sistema
docker images

# Filtrar por el nombre del proyecto
docker images | grep car-price
```

### 3️⃣ Ejecutar el Contenedor

```bash
# Ejecutar en segundo plano en el puerto 8000
docker run -d -p 8000:8000 --name car_predictor_app car-price-predictor:v1

# Ejecutar con reinicio automático
docker run -d -p 8000:8000 --name car_predictor_app --restart always car-price-predictor:v1

# Ejecutar en modo interactivo para ver logs directamente en la consola
docker run -p 8000:8000 --name car_predictor_app car-price-predictor:v1
```

---

## 📊 5. Ver Logs y Diagnóstico de Docker

```bash
# Ver logs completos
docker logs car_predictor_app

# Ver logs en tiempo real (modo seguimiento)
docker logs -f car_predictor_app

# Ver últimas 20 líneas
docker logs --tail 20 car_predictor_app

# Ver logs con marcas de tiempo (timestamps)
docker logs -t car_predictor_app
```

---

## 🔧 6. Gestión del Contenedor

```bash
# Ver contenedores activos
docker ps

# Ver todos los contenedores (incluyendo detenidos)
docker ps -a

# Detener el contenedor
docker stop car_predictor_app

# Iniciar el contenedor detenido
docker start car_predictor_app

# Reiniciar el contenedor
docker restart car_predictor_app

# Pausar procesos del contenedor
docker pause car_predictor_app

# Reanudar contenedor pausado
docker unpause car_predictor_app

# Ver estado de ejecución
docker inspect -f '{{.State.Status}}' car_predictor_app
```

---

## 📈 7. Monitoreo y Estadísticas del Contenedor

```bash
# Ver uso de CPU, memoria y red en tiempo real
docker stats car_predictor_app

# Ver procesos activos dentro del contenedor
docker top car_predictor_app

# Ver configuración detallada del contenedor
docker inspect car_predictor_app

# Ver la IP asignada al contenedor
docker inspect -f '{{.NetworkSettings.IPAddress}}' car_predictor_app

# Ver mapeo de puertos
docker port car_predictor_app
```

---

## 🐚 8. Comandos Dentro del Contenedor

```bash
# Ver archivos en el directorio de trabajo del contenedor
docker exec car_predictor_app ls -la /app

# Ver modelos cargados dentro del contenedor
docker exec car_predictor_app ls -la /app/models

# Probar el endpoint de salud desde adentro
docker exec car_predictor_app curl http://localhost:8000/health

# Abrir una terminal interactiva (Bash) dentro del contenedor
docker exec -it car_predictor_app bash

# Dentro de bash puedes hacer:
ls -la
cat app_docker.py
pwd
env
exit  # Salir de la terminal del contenedor
```

---

## 📁 9. Copiar Archivos entre Host y Contenedor