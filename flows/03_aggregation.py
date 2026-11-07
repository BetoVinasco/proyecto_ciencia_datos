"""Agregaciones y estadísticas de la capa Gold."""

import pandas as pd


def group_stats(df: pd.DataFrame) -> dict:
    """Genera estadísticas descriptivas agrupadas."""

    tablas = {
        "age_by_survival": (
            df.groupby("Survived")["Age"]
            .describe()
            .reset_index()
        ),

        "survival_by_sex": (
            df.groupby("Sex")["Survived"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        ),

        "survival_by_pclass": (
            df.groupby("Pclass")["Survived"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        ),

        "survival_by_alone": (
            df.groupby("Alone")["Survived"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        ),
    }

    return tablas