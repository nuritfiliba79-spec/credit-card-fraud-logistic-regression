"""
Logistic Regression for credit-card fraud detection (imbalanced data).

- Random stratified 80/20 train/test split
- Imbalance handled with class_weight="balanced"
- Metrics: Recall, Precision, F1, ROC AUC (+ PR AUC)
- Writes data.js, consumed by index.html (the dashboard)
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "credit_card_fraud_10k (1).csv"
SEED = 42

NUMERIC = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]
CATEGORICAL = ["merchant_category"]
TARGET = "is_fraud"


def build_model(class_weight):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    clf = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    # Random 80/20 split; stratify keeps the ~1.5% fraud rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    balanced = build_model("balanced").fit(X_train, y_train)
    baseline = build_model(None).fit(X_train, y_train)

    p_bal = balanced.predict_proba(X_test)[:, 1]
    p_base = baseline.predict_proba(X_test)[:, 1]

    m_bal = metrics(y_test, p_bal)
    m_base = metrics(y_test, p_base)

    # Threshold that maximises F1 on the balanced model (reported, not used as default)
    prec, rec, thr = precision_recall_curve(y_test, p_bal)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best_thr = float(thr[np.argmax(f1s)])

    fpr, tpr, _ = roc_curve(y_test, p_bal)
    fpr_b, tpr_b, _ = roc_curve(y_test, p_base)

    # Coefficients (on standardised features) + scaler params for the live predictor
    pre = balanced.named_steps["pre"]
    clf = balanced.named_steps["clf"]
    scaler = pre.named_transformers_["num"]
    cats = list(pre.named_transformers_["cat"].categories_[0])
    names = NUMERIC + [f"merchant_category={c}" for c in cats]
    coefs = [{"feature": n, "coef": float(c)} for n, c in zip(names, clf.coef_[0])]

    out = {
        "dataset": {
            "rows": int(len(df)),
            "fraud": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_fraud": int(y_train.sum()),
            "test_fraud": int(y_test.sum()),
        },
        "balanced": m_bal,
        "baseline": m_base,
        "best_f1_threshold": best_thr,
        "best_f1_metrics": metrics(y_test, p_bal, best_thr),
        "roc": {
            "balanced": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
            "baseline": {"fpr": fpr_b.round(4).tolist(), "tpr": tpr_b.round(4).tolist()},
        },
        "test": {"y": y_test.astype(int).tolist(), "p": p_bal.round(5).tolist()},
        "coefficients": coefs,
        "model": {
            "intercept": float(clf.intercept_[0]),
            "numeric": NUMERIC,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "categories": cats,
        },
    }

    with open("data.js", "w", encoding="utf-8") as f:
        f.write("window.MODEL_DATA = ")
        json.dump(out, f)
        f.write(";\n")

    for name, m in [("balanced", m_bal), ("baseline", m_base)]:
        print(
            f"{name:9s} recall={m['recall']:.3f} precision={m['precision']:.3f} "
            f"f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f} cm={m['cm']}"
        )
    print(f"best F1 threshold={best_thr:.3f} -> {out['best_f1_metrics']}")


if __name__ == "__main__":
    main()
