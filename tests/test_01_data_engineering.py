
import importlib

import pandas as pd
import pytest

# Importar el módulo de ingeniería de datos
ing = importlib.import_module("flows.01_data_engineering")


def sample_df():
    """Crear un DataFrame de prueba con las columnas del Titanic."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3],
        "Survived": [1, 0, 1],
        "Pclass": [1, 3, 2],
        "Name": ["Passenger 1", "Passenger 2", "Passenger 3"],
        "Sex": ["female", "male", "male"],
        "Age": [25.0, 30.0, 10.0],
        "SibSp": [0, 1, 0],
        "Parch": [0, 0, 1],
        "Ticket": ["A1", "A2", "A3"],
        "Fare": [50.0, 10.0, 20.0],
        "Cabin": [None, None, "C1"],
        "Embarked": ["S", "C", "Q"],
    })


def test_validate_bronze_accepts_valid_dataframe():
    df = sample_df()

    result = ing.validate_bronze(df)

    assert list(result.columns) == ing.EXPECTED_COLUMNS


def test_validate_bronze_rejects_missing_columns():
    df = sample_df().drop(columns=["Age"])

    with pytest.raises(ValueError, match="Faltan columnas"):
        ing.validate_bronze(df)


def test_validate_bronze_rejects_duplicate_ids():
    df = sample_df()
    df.loc[1, "PassengerId"] = 1

    with pytest.raises(ValueError, match="duplicados"):
        ing.validate_bronze(df)


def test_cast_types_converts_numeric_columns():
    df = sample_df()

    result = ing.cast_types(df)

    assert result["Survived"].dtype == "int64"
    assert result["Pclass"].dtype == "int64"
    assert result["Age"].dtype == "float64"
    assert result["Fare"].dtype == "float64"


def test_add_person_classifies_passengers():
    df = sample_df()

    result = ing.add_person(df)

    assert result["Person"].tolist() == [
        "female",
        "male",
        "child"
    ]
