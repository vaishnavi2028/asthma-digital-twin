"""Shared helpers - patient-level split, model fitting, honest metrics."""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

P = dict(n_estimators=600, learning_rate=0.06, num_leaves=31, min_child_samples=100, subsample=0.7,
         subsample_freq=1, colsample_bytree=0.8, max_bin=63, n_jobs=1, verbose=-1)


def patient_split(df, seed=42, frac=(0.6, 0.2)):
    """Split by PATIENT, never by row: one patient's hours must not appear in both train and test."""
    ids = np.array(sorted(df.patient_id.unique()))
    np.random.default_rng(seed).shuffle(ids)
    a, b = int(frac[0] * len(ids)), int((frac[0] + frac[1]) * len(ids))
    return set(ids[:a]), set(ids[a:b]), set(ids[b:])        # train, validation, test


def fit_lgbm(Xtr, ytr, Xva, yva, seed=42):
    m = lgb.LGBMClassifier(random_state=seed, **P)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def threshold_at_spec(y, score, spec=0.90):
    """Alert threshold chosen on VALIDATION data so that at most (1-spec) of calm windows trigger an alert."""
    return float(np.quantile(score[y == 0], spec, method="higher"))


def report(y, score, thr):
    alert = score > thr
    return dict(roc_auc=roc_auc_score(y, score), pr_auc=average_precision_score(y, score),
                sensitivity=float(alert[y == 1].mean()), specificity=float(1 - alert[y == 0].mean()))