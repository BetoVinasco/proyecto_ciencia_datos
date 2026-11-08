"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Genera cuatro tablas estadísticas descriptivas con groupby."""
    age_by_survival = df.groupby("Survived")["Age"].describe().reset_index()
    age_by_survival.columns = [str(c) for c in age_by_survival.columns]

    survival_by_sex = df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    survival_by_sex.columns = [str(c) for c in survival_by_sex.columns]

    survival_by_pclass = df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    survival_by_pclass.columns = [str(c) for c in survival_by_pclass.columns]

    survival_by_alone = df.groupby("Alone")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    survival_by_alone.columns = [str(c) for c in survival_by_alone.columns]

    return {
        "age_by_survival": age_by_survival,
        "survival_by_sex": survival_by_sex,
        "survival_by_pclass": survival_by_pclass,
        "survival_by_alone": survival_by_alone,
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Devuelve {'chi2': float, 'p_value': float, 'dof': int}."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p_val, dof, _ = stats.chi2_contingency(tabla)
    return {
        "chi2": float(chi2),
        "p_value": float(p_val),
        "dof": int(dof),
    }


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia cruzando Pclass y Sex."""
    pt = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
    pt = pt.rename(columns=str).reset_index()
    pt.columns = [str(c) for c in pt.columns]
    return pt


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa y número de pasajeros por puerto de embarque."""
    res = df.groupby("Embarked")["Survived"].agg(["mean", "count"]).reset_index()
    res.columns = [str(c) for c in res.columns]
    return res


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por puerto de embarque estratificada por clase."""
    pt = df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    pt = pt.rename(columns=str).reset_index()
    pt.columns = [str(c) for c in pt.columns]
    return pt


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Crea df['FareQ'] con cuartiles de Fare calculados dentro de cada Pclass."""
    df = df.copy()
    df["FareQ"] = df.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return df


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia según cuartil de tarifa estratificado por clase."""
    df_q = add_fare_quartile(df)
    pt = df_q.pivot_table(values="Survived", index="FareQ", columns="Pclass", aggfunc="mean")
    pt = pt.rename(columns=str).reset_index()
    pt.columns = [str(c) for c in pt.columns]
    return pt


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia y cantidad de pasajeros por tamaño de familia."""
    res = df.groupby("Familia")["Survived"].agg(["mean", "count"]).reset_index()
    res.columns = [str(c) for c in res.columns]
    return res


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Identifica valores atípicos mediante el criterio del rango intercuartílico (IQR)."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume los atípicos en Fare, Age y Familia, y compara la supervivencia de atípicos de Fare."""
    age_col = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"
    fare_valid = df["Fare"].dropna()
    age_valid = df[age_col].dropna()
    fam_valid = df["Familia"].dropna()

    fare_outliers = outlier_mask_iqr(fare_valid)
    age_outliers = outlier_mask_iqr(age_valid)
    fam_outliers = outlier_mask_iqr(fam_valid)

    # Identificar índices de atípicos de Fare
    outlier_idx = fare_valid[fare_outliers].index
    non_outlier_idx = fare_valid[~fare_outliers].index

    surv_outliers = float(df.loc[outlier_idx, "Survived"].mean()) if len(outlier_idx) > 0 else 0.0
    surv_non_outliers = float(df.loc[non_outlier_idx, "Survived"].mean()) if len(non_outlier_idx) > 0 else 0.0

    res = pd.DataFrame([
        {"variable": "Fare", "n_outliers": int(fare_outliers.sum()), "outlier_survival_rate": surv_outliers},
        {"variable": "Age", "n_outliers": int(age_outliers.sum()), "outlier_survival_rate": None},
        {"variable": "Familia", "n_outliers": int(fam_outliers.sum()), "outlier_survival_rate": None},
        {"variable": "Fare_NonOutliers", "n_outliers": int((~fare_outliers).sum()), "outlier_survival_rate": surv_non_outliers},
    ])
    res.columns = [str(c) for c in res.columns]
    return res


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table_to_save = table.copy()
        table_to_save.columns = [str(c) for c in table_to_save.columns]
        table_to_save.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(
        json.dumps(chi2_sex_survival(df), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
