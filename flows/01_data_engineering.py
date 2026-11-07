"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE, BRONZE_LABELED_NAME, SILVER, SILVER_FINAL_NAME, SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


#def validate_bronze(df: pd.DataFrame) -> pd.DataFrame: ...      # contrato de entrada
def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Valida que Bronze tenga las columnas esperadas y sin IDs duplicados."""

    missing = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    if df["PassengerId"].duplicated().any():
        raise ValueError("PassengerId contiene valores duplicados")

    return df

#def cast_types(df: pd.DataFrame) -> pd.DataFrame: ...           # Survived -> int, etc.
def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas a sus tipos de datos correspondientes."""

    out = df.copy()

    # Variables enteras
    for col in ["Survived", "Pclass", "SibSp", "Parch"]:
        out[col] = out[col].astype(int)

    # Variables decimales
    for col in ["Age", "Fare"]:
        out[col] = out[col].astype(float)

    return out




#def add_person(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 8
def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica a los pasajeros como child, female o male."""

    out = df.copy()

    out["Person"] = np.where(
        out["Age"] < 16,
        "child",
        out["Sex"]
    )

    return out


#def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame: ...   # Pasos 11 y 24 (TCabin, HasCabin)
def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la cubierta de la cabina y crea indicador de disponibilidad."""

    out = df.copy()

    # Extraer la primera letra de Cabin.
    # Si no existe información, asignar X (desconocida).
    out["TCabin"] = out["Cabin"].str[0].fillna("X")

    # Indicar si el pasajero tiene una cabina registrada.
    # 1 = Tiene cabina, 0 = No tiene cabina.
    out["HasCabin"] = out["Cabin"].notna().astype(int)

    return out


#def add_family_features(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 14 (Familia, Alone)
def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el tamaño de familia y clasifica si el pasajero viajaba solo."""

    out = df.copy()

    # Número de familiares a bordo
    out["Familia"] = out["SibSp"] + out["Parch"]

    # Clasificación del pasajero
    out["Alone"] = np.where(
        out["Familia"] > 0,
        "with Family",
        "Alone"
    )

    return out


#def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 16 (Survivor)
def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega Survivor mapeando 0 -> No y 1 -> Yes."""

    out = df.copy()

    out["Survivor"] = out["Survived"].map({
        0: "No",
        1: "Yes"
    })

    return out

#def add_title(df: pd.DataFrame) -> pd.DataFrame: ...            # Paso 25 (TTitulo)
def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título del nombre del pasajero."""
    out = df.copy()
    patron = r',\s*([^.]+)\.'
    out["TTitulo"] = out["Name"].str.extract(
        patron, expand=False
    ).str.strip()
    return out

def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 26


#def impute_embarked(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 21
def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa los valores faltantes de Embarked utilizando la moda."""
    out = df.copy()
    moda = out["Embarked"].mode()[0]
    out["Embarked"] = out["Embarked"].fillna(moda)
    return out

#def impute_age(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 22
def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa Age con el promedio por Sex.

    Conserva la edad original en AgeOriginal y marca las filas
    imputadas en AgeImputada.
    """
    out = df.copy()

    # 1. Conservar la edad original y marcar los valores faltantes
    out["AgeOriginal"] = out["Age"]
    out["AgeImputada"] = out["Age"].isna()

    # 2. Calcular el promedio de edad por género
    promedios = out.groupby("Sex")["Age"].mean()

    # 3. Rellenar las edades faltantes con el promedio de su género
    out["Age"] = out["Age"].fillna(out["Sex"].map(promedios))

    # 4. Registrar la desviación estándar antes y después
    log.info("Desviación estándar de Age antes de imputar: %.2f",
             out["AgeOriginal"].std())

    log.info("Desviación estándar de Age después de imputar: %.2f",
             out["Age"].std())

    return out

#def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame: ...  # Pasos 21, 23, 24, 26
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica las variables categóricas en columnas numéricas."""

    out = df.copy()

    out["NEmbarked"] = out["Embarked"].map({
        "S": 0,
        "C": 1,
        "Q": 2
    })

    out["NSex"] = out["Sex"].map({
        "female": 0,
        "male": 1
    })

    mapa = {letra: i for i, letra in enumerate(DECKS)}
    out["NCabin"] = out["TCabin"].map(mapa)

    return out

def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 32
def validate_silver(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 33


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME