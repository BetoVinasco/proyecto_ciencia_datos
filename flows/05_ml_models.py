"""Modelado: Silver -> modelo entrenado y guardado."""
from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import MODELS, SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Familia"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "TTitulo", "Alone"]
CABIN_NUM = ["HasCabin"]            # features "sospechosas": se evalúan en el Paso 61
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# --- Decisiones congeladas (resultado del cuaderno 04; justificadas en reports/problema_ml.md) ---
MODEL_NAME = "titanic_model_v1"
FINAL_ESTIMATOR = "hgb"                 # nombre en get_candidates()
FINAL_PARAMS = {                        # mejores hiperparámetros encontrados en Paso 57
    "model__learning_rate": 0.05,
    "model__max_depth": 3,
    "model__max_iter": 200,
    "model__l2_regularization": 1.0,
}
USE_CABIN_FEATURES = False              # decisión del Paso 61: cabina se excluye por artefacto de registro
THRESHOLD = 0.5                         # decisión del Paso 53


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver. Usa la edad ORIGINAL (con NaN): el modelo imputa por su cuenta."""
    df = pd.read_parquet(Path(path) if path else SILVER / SILVER_NAME)
    X = df[CANDIDATE_COLUMNS].copy()
    if "AgeOriginal" in df.columns:
        X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Devuelve X_train, X_test, y_train, y_test, estratificado por y."""
    ts = settings.test_size if test_size is None else test_size
    rs = settings.random_state if random_state is None else random_state
    return train_test_split(X, y, test_size=ts, random_state=rs, stratify=y)


class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Rellena 'Age' con el promedio de su grupo, aprendido SOLO en fit()."""

    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        cols = list(self.group_cols)
        self.group_means_ = X.groupby(cols)[self.target].mean()
        self.global_mean_ = X[self.target].mean()
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.group_cols)
        means = self.group_means_.rename("_fill").reset_index()
        fill = X[cols].merge(means, on=cols, how="left")["_fill"].to_numpy(dtype=float)
        fill = np.where(np.isnan(fill), self.global_mean_, fill)
        X[self.target] = X[self.target].fillna(pd.Series(fill, index=X.index))
        return X


def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """GroupAgeImputer -> ColumnTransformer -> modelo, todo en un Pipeline."""
    num_features = NUM_FEATURES if num_features is None else num_features
    cat_features = CAT_FEATURES if cat_features is None else cat_features

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
    ])
    prep = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        sparse_threshold=0,
    )
    return Pipeline([
        ("age_imputer", GroupAgeImputer(group_cols=("Sex", "Pclass"))),
        ("prep", prep),
        ("model", model),
    ])


def get_candidates(random_state: int = 42) -> dict:
    """Diccionario {nombre: estimador sin entrenar}."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }


def cross_validate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    num_features: list[str] | None = None,
    cat_features: list[str] | None = None,
    random_state: int | None = None,
    cv_folds: int | None = None,
) -> pd.DataFrame:
    """Devuelve un DataFrame con media y desviación de cada métrica por modelo."""
    rs = settings.random_state if random_state is None else random_state
    folds = settings.cv_folds if cv_folds is None else cv_folds
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)

    candidates = get_candidates(rs)
    results = []
    for name, model in candidates.items():
        pipe = build_pipeline(model, num_features=num_features, cat_features=cat_features)
        cv_res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
        rec = {
            "model": name,
            "test_roc_auc_mean": float(cv_res["test_roc_auc"].mean()),
            "test_roc_auc_std": float(cv_res["test_roc_auc"].std()),
            "train_roc_auc_mean": float(cv_res["train_roc_auc"].mean()),
            "test_accuracy_mean": float(cv_res["test_accuracy"].mean()),
            "test_f1_mean": float(cv_res["test_f1"].mean()),
            "test_precision_mean": float(cv_res["test_precision"].mean()),
            "test_recall_mean": float(cv_res["test_recall"].mean()),
            "auc_overfit_gap": float(cv_res["train_roc_auc"].mean() - cv_res["test_roc_auc"].mean()),
        }
        results.append(rec)
        log.info("CV %s - ROC-AUC: %.4f (+/- %.4f)", name, rec["test_roc_auc_mean"], rec["test_roc_auc_std"])
    df_res = pd.DataFrame(results).sort_values(by=f"test_{PRIMARY_METRIC}_mean", ascending=False).reset_index(drop=True)
    return df_res


def get_param_grids() -> dict[str, dict]:
    """Rejillas por modelo, con los nombres que espera el Pipeline (prefijo 'model__')."""
    return {
        "logreg": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        },
        "forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "hgb": {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__max_iter": [100, 200],
            "model__l2_regularization": [0.0, 1.0],
        },
    }


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = THRESHOLD,
) -> dict:
    """Devuelve exactitud, F1, ROC-AUC y el tamaño de la prueba (dict serializable a JSON)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "threshold": float(threshold),
        "n_test": len(y_test),
    }


def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Construye el pipeline con las decisiones congeladas y lo entrena con TODO (X, y)."""
    num = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    cat = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    base_model = get_candidates(settings.random_state)[FINAL_ESTIMATOR]
    pipe = build_pipeline(base_model, num_features=num, cat_features=cat)
    pipe.set_params(**FINAL_PARAMS)
    return pipe.fit(X, y)


def save_model(
    pipeline: Pipeline,
    metadata: dict,
    name: str = MODEL_NAME,
    models_dir: Path = MODELS,
) -> Path:
    """Guarda el pipeline (.joblib) y un archivo de metadatos (.metadata.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "model_sha256": sha256_of(model_path),
    }
    (models_dir / f"{name}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info("Modelo guardado en %s", model_path)
    return model_path


def load_model(path: Path) -> tuple[Pipeline, dict]:
    """Carga el pipeline y sus metadatos; avisa si la versión de scikit-learn cambió."""
    path = Path(path)
    pipeline = joblib.load(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning(
            "Modelo entrenado con scikit-learn %s; versión actual %s",
            meta.get("sklearn_version"),
            sklearn.__version__,
        )
    return pipeline, meta


def predict(pipeline: Pipeline, X: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Devuelve un DataFrame con 'proba_survive' y 'pred' (0/1) para cada fila de X."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame({
        "proba_survive": proba,
        "pred": pred,
    }, index=X.index)


def run(in_dir: Path | None = None, models_dir: Path = MODELS) -> Path:
    """Silver -> evaluación honesta -> modelo final entrenado y guardado."""
    silver_path = (in_dir or SILVER) / SILVER_NAME
    X, y = load_dataset(silver_path)
    X_tr, X_te, y_tr, y_te = split(X, y)

    # 1. Evaluación honesta sobre test
    num = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    cat = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    eval_model = get_candidates(settings.random_state)[FINAL_ESTIMATOR]
    pipe_eval = build_pipeline(eval_model, num_features=num, cat_features=cat)
    pipe_eval.set_params(**FINAL_PARAMS)
    pipe_eval.fit(X_tr, y_tr)
    metrics = evaluate_on_test(pipe_eval, X_te, y_te, threshold=THRESHOLD)
    log.info(
        "Métricas en Test: Accuracy=%.4f, F1=%.4f, ROC-AUC=%.4f (n=%d)",
        metrics["accuracy"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["n_test"],
    )

    # 2. Entrenar modelo final con todos los datos etiquetados
    final_pipeline = train_final(X, y)

    # 3. Guardar con metadatos
    metadata = {
        "estimator": FINAL_ESTIMATOR,
        "params": FINAL_PARAMS,
        "features": {
            "num_features": num,
            "cat_features": cat,
        },
        "target": TARGET,
        "threshold": THRESHOLD,
        "use_cabin_features": USE_CABIN_FEATURES,
        "random_state": settings.random_state,
        "n_train": len(X),
        "data_sha256": sha256_of(silver_path),
        "test_metrics": metrics,
    }
    model_path = save_model(final_pipeline, metadata, name=MODEL_NAME, models_dir=models_dir)
    return model_path
