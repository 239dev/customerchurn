"""Train/test split, logistic regression baseline, tuned XGBoost."""
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier

from .features import build_preprocessor


def split(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def fit_logreg(X_train, y_train):
    pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_xgb(X_train, y_train, cv_splits=5):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1,
        )),
    ])
    param_grid = {
        "clf__max_depth": [3, 4, 6],
        "clf__min_child_weight": [1, 5],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    # n_jobs=-1 here AND on the estimator crashes worker processes on Windows,
    # so keep the parallelism on the search only.
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search


def evaluate(model, X_test, y_test, name="model", threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    report = classification_report(y_test, pred, target_names=["Stay", "Churn"], output_dict=True)
    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    print(f"\n{name} (threshold={threshold})")
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print("ROC-AUC:", round(auc, 4))
    print("PR-AUC :", round(pr_auc, 4))
    print("Confusion matrix:\n", cm)

    return {
        "proba": proba, "auc": auc, "pr_auc": pr_auc,
        "recall_churn": report["Churn"]["recall"],
        "precision_churn": report["Churn"]["precision"],
        "cm": cm,
    }
