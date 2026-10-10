"""Baselines + first ablation on ONE patient-level split (Day 6 makes it rigorous)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import EHR_COLS, ENV_COLS, WEAR_COLS, TIME_COLS
from modeling import patient_split, fit_lgbm, threshold_at_spec, report

INHALER = ["puffs_6h", "puffs_24h", "puffs_72h", "hours_since_puff"]
PHYSIO = [c for c in WEAR_COLS if c not in INHALER]            # heart rate, breathing, SpO2, steps, sleep

df = pd.read_csv("data/processed/model_table.csv.gz")
df = df[df.eligible].reset_index(drop=True)
tr_p, va_p, te_p = patient_split(df)
tr, va, te = (df[df.patient_id.isin(s)] for s in (tr_p, va_p, te_p))
print(f"patients train/val/test: {len(tr_p)}/{len(va_p)}/{len(te_p)} | rows {len(tr):,}/{len(va):,}/{len(te):,}")
print(f"flare rate train/val/test: {tr.label.mean():.3f}/{va.label.mean():.3f}/{te.label.mean():.3f}")

rows = []
def add(name, kind, s_va, s_te):
    thr = threshold_at_spec(va.label.values, s_va, 0.90)
    rows.append(dict(model=name, kind=kind, **report(te.label.values, s_te, thr)))

# ---- rules (no learning): what a simple alarm would do ----
add("Rule: puffs in last 24h", "rule", va.puffs_24h.values, te.puffs_24h.values)
add("Rule: PM2.5 24h mean", "rule", va.pm25_mean_24h.values, te.pm25_mean_24h.values)

# ---- simple linear model on everything ----
allc = EHR_COLS + ENV_COLS + WEAR_COLS + TIME_COLS
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[allc], tr.label)
add("Logistic regression (all)", "linear", lr.predict_proba(va[allc])[:, 1], lr.predict_proba(te[allc])[:, 1])

# ---- the ablation: which data stream adds what? ----
groups = {
    "EHR only": EHR_COLS,
    "Inhaler history only": INHALER + TIME_COLS,
    "Wearables (no inhaler)": PHYSIO + TIME_COLS,
    "Wearables + inhaler": WEAR_COLS + TIME_COLS,
    "Wearables + inhaler + air": WEAR_COLS + ENV_COLS + TIME_COLS,
    "EHR + wearables + inhaler + air (fused)": EHR_COLS + WEAR_COLS + ENV_COLS + TIME_COLS,
}
for name, cols in groups.items():
    m = fit_lgbm(tr[cols], tr.label, va[cols], va.label)
    add(name, "lightgbm", m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1])
    print(f"fitted {name} ({len(cols)} features, {m.best_iteration_} trees)", flush=True)

res = pd.DataFrame(rows)
res.to_csv("results/baselines.csv", index=False)
print("\n", res.drop(columns="kind").round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.2))
y = np.arange(len(res))[::-1]
ax.barh(y + 0.2, res.pr_auc, 0.4, label="PR-AUC", color="#d9480f")
ax.barh(y - 0.2, res.roc_auc, 0.4, label="ROC-AUC", color="#4c78a8")
ax.set_yticks(y); ax.set_yticklabels(res.model, fontsize=8); ax.axvline(df.label.mean(), color="k", ls=":", lw=.8)
ax.legend(frameon=False); ax.set_title("Baselines and ablation (single patient-level split)")
plt.tight_layout(); plt.savefig("results/baselines.png", dpi=140)