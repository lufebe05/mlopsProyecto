# 🚗 Predicción de Precios de Vehículos Usados - MLOps Pipeline

Este proyecto implementa una solución completa de **Machine Learning Operations (MLOps)** de extremo a extremo (End-to-End) para la estimación de precios comerciales de vehículos usados. El flujo abarca desde la ingesta de datos, experimentación y tracking con **MLflow**, orquestación con **Prefect**, optimización de hiperparámetros con **Optuna**, hasta el empaquetado y despliegue local mediante una API REST en **FastAPI** con interfaz web servida vía contenedores **Docker**.

---

## 📌 Hipótesis y Planteamiento del Problema de Negocio

En el mercado de compra y venta de automóviles usados, la fijación incorrecta de precios genera fricción: 
- **Sobreprecio:** Aumenta el tiempo que el vehículo pasa en inventario sin venderse, generando costos de depreciación y almacenamiento.
- **Subvaloración:** Ocasiona pérdidas directas de margen de ganancia para particulares o concesionarios.

**Objetivo de la solución:**
Desarrollar un modelo de regresión supervisado que prediga con precisión el precio comercial estimado de reventa (`selling_price`) en función de atributos del vehículo como año, kilometraje, tipo de combustible, transmisión y tipo de vendedor, exponiendo la predicción a través de un servicio web interactivo para facilitar la toma de decisiones comerciales.

---

## 📂 Estructura del Proyecto

La estructura del repositorio sigue las mejores prácticas de modularidad, trazabilidad y separación de responsabilidades:

```text
mlopsProyecto/
├── .github/
│   └── workflows/          # Flujos de integración continua (CI/CD)
├── app/                    # Aplicación de despliegue e inferencia
│   ├── static/             # Archivos estáticos de frontend (CSS, JavaScript, imágenes)
│   ├── templates/          # Vistas HTML para la interfaz web de usuario
│   ├── main.py             # Servicio web API REST con FastAPI
│   └── Dockerfile          # Contenedor para despliegue local reproducible
├── data/                   # Almacenamiento local de conjuntos de datos
│   ├── raw/                # Datos originales sin procesar (carData.csv)
│   └── processed/          # Datos limpios y listos para modelado
├── docs/                   # Bitácora e informe técnico del proyecto
├── models/                 # Modelos serializados y artefactos de inferencia
├── notebooks/              # Cuadernos Jupyter para EDA y pruebas exploratorias
├── pipelines/              # Flujos de orquestación con Prefect
├── src/                    # Código fuente modular y reutilizable
│   ├── data/               # Scripts de carga, descarga y validación de datos
│   ├── features/           # Pipelines de preprocesamiento e ingeniería de variables
│   └── models/             # Lógica de entrenamiento, evaluación y tuning con Optuna
├── tests/                  # Pruebas unitarias para pipelines y endpoints
├── .dockerignore           # Exclusiones para el empaquetado en Docker
├── .gitignore              # Archivos y carpetas ignorados por el control de versiones
├── pyproject.toml          # Definición del entorno y dependencias del proyecto
└── README.md               # Documentación general del repositorio