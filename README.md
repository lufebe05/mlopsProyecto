# 🚗 Predicción de Precios de Vehículos Usados - MLOps Pipeline

Este proyecto implementa una solución completa de **Machine Learning Operations (MLOps)** de extremo a extremo (End-to-End) para la estimación de precios comerciales de vehículos usados. El flujo abarca desde la ingesta de datos, experimentación y tracking con **MLflow**, orquestación con **Prefect**, optimización de hiperparámetros con **Optuna**, hasta el empaquetado y despliegue local mediante una API REST en **FastAPI** con interfaz web servida vía contenedores **Docker**.

* **Integrante 1:** Alejandro Ramirez Marin
* **Integrante 2:** Luis Fernando Betancurt 
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
```
---

## 🛠️ Stack Tecnológico

| Componente | Herramienta | Rol en el Proyecto |
| :--- | :--- | :--- |
| **Entorno y Dependencias** | Python 3.11+, `uv` | Gestión determinista de paquetes y entornos virtuales |
| **Procesamiento y Modelado** | scikit-learn, pandas, numpy | Pipelines de transformación, regresión y métricas de negocio |
| **Tracking de Experimentos** | MLflow (backend SQLite) | Registro estructurado de parámetros, métricas y artefactos serializados |
| **Optimización de Hiperparámetros** | Optuna | Búsqueda bayesiana (TPE) con validación cruzada anidada |
| **Servicio e Inferencia** | FastAPI, Jinja2, Uvicorn | Endpoint REST y renderizado del formulario web para cotización |
| **Contenedorización** | Docker | Empaquetado portable y reproducible de la solución web |
| **Control de Versiones** | Git | Trazabilidad del código fuente |

---

## 🔬 Flujo de Modelado y Resultados

El proyecto evalúa modelos lineales y ensambles no lineales transformando el objetivo comercial mediante $\log(1 + y)$ para estabilizar la varianza, reportando métricas tanto en la escala logarítmica como en la escala real de rupias indias (INR):

### Tabla Comparativa de Desempeño

| Iteración | Modelo | Test RMSE (INR) | Test MAE (INR) | Test $R^2$ (Real) | Test $R^2$ (Log) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01 (Baseline)** | Ridge Regression | 377,865.46 | 157,680.44 | 0.5568 | 0.7574 |
| **02 (Ensamble)** | Random Forest Regressor | 327,739.11 | 145,244.37 | 0.6666 | 0.7710 |
| **03 (Optimizado)** | **Random Forest (Tuned)** | **328,680.44** | **145,388.37** | **0.6646** | **0.7708** |

### Hallazgos de la Optimización Bayesiana (Optuna)
- **Mejor Trial:** `Trial_23` (CV RMSE: 273,029.58 INR con validación cruzada $k=5$).
- **Configuración ganadora:** `n_estimators: 150`, `max_depth: 20`, `min_samples_split: 8`, `min_samples_leaf: 1`, `max_features: 0.92`.
- **Conclusión técnica:** El ensamble captura adecuadamente las no-linealidades observadas durante el análisis exploratorio frente al modelo lineal, logrando un balance que previene el sobreajuste.

---

## 📦 Artefactos Generados

- `models/preprocessor.joblib`: Objeto `ColumnTransformer` ajustado para imputación, codificación One-Hot de variables categóricas, escalado estándar y homologación de marcas vehiculares.
- `models/random_forest_tuned.joblib`: Modelo final entrenado y calibrado con los hiperparámetros óptimos.
- `mlflow.db`: Base de datos SQLite local que contiene el historial completo de corridas, métricas comparativas y artefactos de experimentación.

---

## ⚡ Guía de Ejecución

Para consultar el paso a paso detallado con los comandos exactos listos para copiar y ejecutar (instalación, entrenamiento, interfaz visual de MLflow y despliegue local o con Docker), consulta el manual:

👉 **[Guia_ejecucion.md](./guia_ejecucion.md)**