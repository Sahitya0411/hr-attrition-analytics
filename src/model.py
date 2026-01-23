"""Attrition risk models: logistic regression and XGBoost.

Both models share one preprocessing step and handle class imbalance
(about 16% leavers): logistic regression uses `class_weight="balanced"`,
XGBoost uses `scale_pos_weight = negatives / positives`.

Per-employee explanations are exact additive contributions to the log-odds:
- Logistic regression: coef * (x - training mean), in the scaled space.
- XGBoost: TreeSHAP values from `pred_contribs=True`, built into XGBoost.
One-hot columns are summed back to their original feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: list[str] = [
    "Age", "DistanceFromHome", "Education", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion", "WorkLifeBalance",
    "PerformanceRating", "EnvironmentSatisfaction", "JobInvolvement", "StockOptionLevel",
    "NumCompaniesWorked", "YearsInCurrentRole", "YearsWithCurrManager",
    "RelationshipSatisfaction", "TrainingTimesLastYear", "PercentSalaryHike",
]
CATEGORICAL_FEATURES: list[str] = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "AttritionFlag"

LOGREG, XGBOOST = "Logistic Regression", "XGBoost"
RANDOM_STATE = 42


@dataclass
class ModelBundle:
    """Everything the Risk Model tab needs, trained once and cached."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    confusion: dict[str, np.ndarray]
    roc: dict[str, tuple[np.ndarray, np.ndarray, float]]
    importance: dict[str, pd.DataFrame]
    defaults: dict[str, object]
    feature_map: list[str]            # transformed column -> original feature
    train_mean: np.ndarray            # mean of transformed training matrix
    n_train: int = 0
    n_test: int = 0
    positive_rate: float = 0.0
    cv_auc: dict[str, tuple[float, float]] = field(default_factory=dict)


def _available(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Feature lists restricted to columns present in df."""
    return [c for c in NUMERIC_FEATURES if c in df], [c for c in CATEGORICAL_FEATURES if c in df]


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def make_models(pos_weight: float, num: list[str], cat: list[str]) -> dict[str, Pipeline]:
    """Unfitted pipelines for both models."""
    return {
        LOGREG: Pipeline([
            ("prep", make_preprocessor(num, cat)),
            ("clf", LogisticRegression(class_weight="balanced", C=0.5, max_iter=5000)),
        ]),
        XGBOOST: Pipeline([
            ("prep", make_preprocessor(num, cat)),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                scale_pos_weight=pos_weight, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=2,
            )),
        ]),
    }


def _feature_map(prep: ColumnTransformer, num: list[str], cat: list[str]) -> list[str]:
    """Map every transformed column back to the original feature name."""
    mapping = list(num)
    encoder: OneHotEncoder = prep.named_transformers_["cat"]
    for feat, cats in zip(cat, encoder.categories_):
        mapping.extend([feat] * len(cats))
    return mapping


def _contributions(name: str, pipe: Pipeline, Xt: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    """Per-row additive log-odds contributions in transformed space (n x p)."""
    clf = pipe.named_steps["clf"]
    if name == LOGREG:
        return (Xt - train_mean) * clf.coef_[0]
    contribs = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
    return contribs[:, :-1]  # last column is the bias term


def _aggregate(contrib: np.ndarray, feature_map: list[str]) -> pd.DataFrame:
    """Sum transformed-column contributions into original features."""
    frame = pd.DataFrame(contrib, columns=feature_map)
    return frame.T.groupby(level=0, sort=False).sum().T


def train_models(df: pd.DataFrame, test_size: float = 0.2) -> ModelBundle:
    """Train both models on a stratified split and compute all evaluation outputs."""
    num, cat = _available(df)
    X, y = df[num + cat], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = make_models(pos_weight, num, cat)

    rows, cms, rocs, importance, cv_auc = [], {}, {}, {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        scores = cross_val_score(make_models(pos_weight, num, cat)[name], X, y, cv=cv, scoring="roc_auc")
        cv_auc[name] = (float(scores.mean()), float(scores.std()))
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc,
            "CV ROC-AUC (5-fold)": scores.mean(),
        })
        cms[name] = confusion_matrix(y_test, pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_test, proba)
        rocs[name] = (fpr, tpr, auc)

    prep_lr = models[LOGREG].named_steps["prep"]
    feature_map = _feature_map(prep_lr, num, cat)
    train_mean = prep_lr.transform(X_train).mean(axis=0)

    # Global importance = mean |contribution| on the test set, per original feature.
    for name, pipe in models.items():
        Xt = pipe.named_steps["prep"].transform(X_test)
        agg = _aggregate(_contributions(name, pipe, Xt, train_mean), feature_map)
        importance[name] = (
            agg.abs().mean().rename("importance").reset_index().rename(columns={"index": "feature"})
            .sort_values("importance", ascending=False, ignore_index=True)
        )

    defaults: dict[str, object] = {c: float(X[c].median()) for c in num}
    defaults.update({c: X[c].mode().iat[0] for c in cat})

    return ModelBundle(
        models=models, metrics=pd.DataFrame(rows).set_index("Model"), confusion=cms, roc=rocs,
        importance=importance, defaults=defaults, feature_map=feature_map, train_mean=train_mean,
        n_train=len(X_train), n_test=len(X_test), positive_rate=float(y.mean()), cv_auc=cv_auc,
    )


def build_input_row(bundle: ModelBundle, values: dict[str, object]) -> pd.DataFrame:
    """One-row frame for prediction: user values over training defaults."""
    row = {**bundle.defaults, **values}
    num = [c for c in NUMERIC_FEATURES if c in row]
    cat = [c for c in CATEGORICAL_FEATURES if c in row]
    return pd.DataFrame([row])[num + cat]


def predict_with_explanation(
    bundle: ModelBundle, name: str, row: pd.DataFrame, top_k: int = 6
) -> tuple[float, pd.DataFrame]:
    """Probability of leaving plus the top contributing factors for one employee."""
    pipe = bundle.models[name]
    proba = float(pipe.predict_proba(row)[:, 1][0])
    Xt = pipe.named_steps["prep"].transform(row)
    contrib = _aggregate(_contributions(name, pipe, Xt, bundle.train_mean), bundle.feature_map).iloc[0]
    out = contrib.rename("contribution").reset_index().rename(columns={"index": "feature"})
    out["value"] = [row.iloc[0][f] for f in out["feature"]]
    out["label"] = [f"{f} = {_fmt(v)}" for f, v in zip(out["feature"], out["value"])]
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top_k)
    return proba, out.reset_index(drop=True)


def risk_band(p: float) -> str:
    """Bucket a probability into a readable band."""
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


def _fmt(v: object) -> str:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:.1f}"
    return str(v)
