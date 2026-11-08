# Predicción de Supervivencia en el Titanic: Buenas Prácticas en Ciencia de Datos e IA

Este repositorio implementa un proyecto integral de Ciencia de Datos y Machine Learning para predecir la supervivencia de los pasajeros del RMS Titanic. El desarrollo aplica estándares rigurosos de ingeniería de software para ciencia de datos: arquitectura Medallion, pipelines de preprocesamiento herméticos (sin fuga de datos), auditoría criptográfica con hashes SHA-256, orquestación modular por etapas, pruebas automatizadas y trazabilidad completa del ciclo de vida del modelo (*Model Card* y serialización en formato `joblib`).

---

## 👥 Autores y Créditos

* **Autor / Desarrollador:** Carlos Alberto Vinasco ([@BetoVinasco](https://github.com/BetoVinasco) — `betovinasco46@gmail.com`)
* **Docente / Diseño del Taller:** Diego Fernando Marín ([@dfmarin](https://github.com/dfmarin) — `dfmarin@gmail.com`)
* **Contexto:** Taller de Buenas Prácticas en Ciencia de Datos e Inteligencia Artificial.

---

## 🎯 Propósito del Proyecto

1. **Formulación del Problema de ML:** Resolver un problema de clasificación binaria supervisada para estimar la probabilidad de supervivencia ($Y \in \{0, 1\}$) de un pasajero individual usando únicamente información disponible al momento del embarque (abril de 1912).
2. **Prevención Estricta de Fuga de Datos (*Data Leakage*):** 
   * Separación honesta de datos de entrenamiento (80%) y prueba (20%) de forma estratificada.
   * La imputación de valores faltantes (como la edad según estrato socioeconómico y sexo con `GroupAgeImputer`) y las transformaciones se calculan exclusivamente sobre el conjunto de entrenamiento mediante `Pipeline` de scikit-learn.
3. **Auditoría y Ablación de Variables:**
   * Análisis crítico sobre la variable `Cabin`: descarte de cabina tras evidenciar un sesgo de registro por supervivencia y un aporte marginal insignificante (+0.0004 ROC-AUC).
   * Exclusión de identificadores redundantes y de alta cardinalidad (`PassengerId`, `Name`, `Ticket`).
4. **Reproducibilidad y Trazabilidad:** Hash SHA-256 de los datos de entrada y del modelo empaquetado, junto con metadatos de entorno (versiones de Python, pandas y scikit-learn) guardados en `titanic_model_v1.metadata.json`.

---

## 🏗️ Arquitectura del Repositorio

El proyecto sigue una estructura limpia, desacoplada y orientada a producción:

```text
├── Makefile                   # Automatización de tareas (instalación, flujos, pruebas, linting)
├── pyproject.toml             # Dependencias del proyecto y configuración de herramientas
├── data/                      # Arquitectura Medallion (gestionada por los flujos)
│   ├── bronze/                # Datos crudos recién ingestados
│   ├── silver/                # Datos limpios, tipados y estructurados (Parquet)
│   └── gold/                  # Vistas agregadas y analíticas de negocio (Parquet)
├── docs/                      # Guías de instalación y material de soporte
│   ├── GUIA.md
│   └── INSTALACION.md
├── flows/                     # Flujos ETL y orquestador del pipeline
│   ├── 00_ingestion.py        # Descarga e ingesta de datos brutos a Bronze
│   ├── 01_data_engineering.py # Transformación y limpieza hacia Silver
│   ├── 03_aggregation.py      # Agregaciones hacia Gold
│   ├── 05_ml_models.py        # Modelado, validación cruzada y serialización
│   ├── common/                # Configuraciones, paths, logging y hashing
│   └── job.py                 # Orquestador general por línea de comandos
├── models/                    # Artefactos del modelo entrenado
│   ├── titanic_model_v1.joblib        # Pipeline completo congelado
│   └── titanic_model_v1.metadata.json # Metadatos de auditoría y métricas
├── notebooks/                 # Exploración interactiva y experimentación
│   ├── 02_exploracion.ipynb   # Análisis exploratorio de datos (EDA)
│   └── 04_modelado.ipynb      # Experimentación y comparación de modelos
├── reports/                   # Documentación técnica y metodológica
│   ├── conclusiones_eda.md    # Hallazgos del análisis exploratorio
│   ├── problema_ml.md         # Formulación del problema y baselines
│   └── model_card.md          # Ficha técnica del modelo en producción
└── tests/                     # Suite de pruebas unitarias automatizadas
    ├── test_00_ingestion.py
    ├── test_01_data_engineering.py
    ├── test_03_aggregation.py
    ├── test_05_ml_models.py
    └── test_job.py
```

---

## 📊 Arquitectura de Datos (Medallion)

* **Bronze (`data/bronze`):** Contiene la copia inmutable de los datos crudos del Titanic (`titanic_bronze.parquet`).
* **Silver (`data/silver`):** Datos con corrección de tipos, nombres limpios, normalización de categorías y conservación de `AgeOriginal` para evitar fugas antes de la partición de modelado.
* **Gold (`data/gold`):** Tablas analíticas agregadas (tasas de supervivencia por clase, grupos de edad, puertos de embarque y estructura familiar).

---

## 🤖 Modelo Seleccionado y Resultados

Tras comparar modelos lineales (Regresión Logística), árboles de decisión, KNN, Random Forest y Boosting mediante validación cruzada estratificada ($k=5$):

* **Estimador Ganador:** `HistGradientBoostingClassifier`
* **Hiperparámetros:**
  * `learning_rate`: 0.1
  * `max_depth`: 4
  * `max_iter`: 200
  * `l2_regularization`: 1.0
* **Umbral de Decisión Operativo:** `threshold = 0.50`
* **Desempeño en Test Honesto ($n = 179$, 20% retenido):**
  * **ROC-AUC:** `0.8238` *(Supera con creces el baseline heurístico de 0.769)*
  * **Exactitud (Accuracy):** `81.01%`
  * **F1-Score:** `0.7424`

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio y crear el entorno virtual

```bash
git clone https://github.com/santiagojimenez0304/proyecto_ciencia_de_datos.git
cd proyecto_ciencia_de_datos

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Ejecutar el Pipeline Completo

Puedes ejecutar todo el ciclo (Ingesta $\rightarrow$ Silver $\rightarrow$ Gold $\rightarrow$ Entrenamiento y Guardado del Modelo) con el comando Makefile:

```bash
make pipeline
```

O paso a paso de forma individual:

```bash
make ingest    # Descarga e ingesta Bronze
make silver    # Transforma y limpia a Silver
make gold      # Genera agregaciones Gold
make train     # Entrena, evalúa y guarda titanic_model_v1.joblib
```

---

## 💡 Cómo Usar el Modelo Serializado (`.joblib`)

El archivo `models/titanic_model_v1.joblib` contiene el pipeline completo empaquetado (transformación de columnas, imputación de edad y clasificador). **No requiere volver a entrenar el modelo ni procesar los datos históricos.**

### Predicción en Python (Script o Notebook)

```python
import joblib
import pandas as pd
from flows.05_ml_models import load_model, predict

# 1. Cargar el pipeline y sus metadatos
pipeline, meta = load_model("models/titanic_model_v1.joblib")

# 2. Datos de nuevos pasajeros
nuevos_pasajeros = pd.DataFrame([
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 211.33,
        "Embarked": "S",
        "TTitulo": "Miss",
        "Alone": "Alone",
    },
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "TTitulo": "Mr",
        "Alone": "with Family",
    }
])

# 3. Realizar predicciones
resultados = predict(pipeline, nuevos_pasajeros, threshold=meta.get("threshold", 0.5))
print(resultados)
```

Salida esperada:
```text
   proba_survive  pred
0       0.963412     1
1       0.082150     0
```

---

## 🧪 Pruebas Unitarias y Calidad de Código

Para garantizar la integridad y confiabilidad del software:

```bash
# Ejecutar suite de pruebas con pytest
make test

# Verificar estilo y buenas prácticas de código con ruff
make lint
```

---

## 📄 Informes y Documentación

Para una lectura profunda sobre cada fase del proyecto, consulta los informes en `reports/`:
* [Conclusiones del Análisis Exploratorio (EDA)](reports/conclusiones_eda.md)
* [Definición del Problema de ML, Baselines y Fuga de Datos](reports/problema_ml.md)
* [Ficha Técnica del Modelo (Model Card)](reports/model_card.md)