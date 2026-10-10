"""Rigorous evaluation, lead time, uncertainty ensemble, and the final model files."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import EHR_COLS, ENV_COLS, WEAR_COLS, TIME_COLS
from modeling import P, fit_lgbm, threshold_at_spec

SEED = 42
FUSED = "Fused (EHR + wearables + air)"
MODELS = {
    "EHR only": EHR_COLS,
    "Wearables + inhaler": WEAR_COLS + TIME_COLS,
    "Wearables + inhaler + air": WEAR_COLS + ENV_COLS + TIME_COLS,
    FUSED: EHR_COLS + WEAR_COLS + ENV_COLS + TIME_COLS,
}
ALL = ["Rule: puffs in last 24h", "Logistic regression (all)"] + list(MODELS)

df = pd.read_csv("data/processed/model_table.csv.gz")
df = df[df.eligible].reset_index(drop=True)
obs = pd.read_csv("data/raw/timeseries.csv.gz")
y, pid, t_idx = df.label.values.astype(int), df.patient_id.values, df.t.values
oof = {n: np.zeros(len(df)) for n in ALL}
thr = {n: {90: np.zeros(len(df)), 95: np.zeros(len(df))} for n in ALL}
rng = np.random.default_rng(SEED)

# ---------------- 1. patient-level 5-fold cross-validation ----------------
demo_pool = None
for fold, (tr, te) in enumerate(GroupKFold(5).split(df, groups=pid)):
    up = np.unique(pid[tr]); rng.shuffle(up)
    is_val = np.isin(pid[tr], up[: int(0.15 * len(up))])          # inner validation patients
    tr_in, va_in = tr[~is_val], tr[is_val]

    def record(name, s_va, s_te):
        oof[name][te] = s_te
        for s in (90, 95):
            thr[name][s][te] = threshold_at_spec(y[va_in], s_va, s / 100)

    record("Rule: puffs in last 24h", df.puffs_24h.values[va_in].astype(float), df.puffs_24h.values[te].astype(float))
    lr_cols = EHR_COLS + ENV_COLS + WEAR_COLS + TIME_COLS
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(df.loc[tr_in, lr_cols], y[tr_in])
    record("Logistic regression (all)", lr.predict_proba(df.loc[va_in, lr_cols])[:, 1], lr.predict_proba(df.loc[te, lr_cols])[:, 1])
    for name, cols in MODELS.items():
        m = fit_lgbm(df.loc[tr_in, cols], y[tr_in], df.loc[va_in, cols], y[va_in], SEED)
        record(name, m.predict_proba(df.loc[va_in, cols])[:, 1], m.predict_proba(df.loc[te, cols])[:, 1])
        if name == FUSED and fold == 0:
            # 5-seed ensemble: the spread between members is our 'model disagreement' signal
            bi, cols_f, members, pv = max(m.best_iteration_ or 40, 30), cols, [], []
            for s in range(5):
                mm = lgb.LGBMClassifier(random_state=100 + s, **{**P, "n_estimators": bi}).fit(df.loc[tr_in, cols], y[tr_in])
                mm.booster_.save_model(f"models/fused_seed{s}.txt")
                pv.append(mm.predict_proba(df.loc[va_in, cols])[:, 1])
            pe = np.mean(pv, axis=0)
            cfg = dict(features=cols, thr90=threshold_at_spec(y[va_in], pe, .90), thr95=threshold_at_spec(y[va_in], pe, .95),
                       n_trees=bi, groups=dict(ehr=EHR_COLS, env=ENV_COLS, wearable=WEAR_COLS))
            demo_pool = np.unique(pid[te])
    print(f"fold {fold} done", flush=True)

# ---------------- 2. flare episodes and lead time ----------------
roll6 = obs.groupby("patient_id").rescue_puffs.transform(lambda s: s.rolling(6).sum()).values
obs["flare"] = roll6 >= 4
episodes = {}
for p, g in obs.groupby("patient_id"):
    f = g.flare.values
    episodes[p] = [k for k in range(30, len(f)) if f[k] and not f[k - 24:k].any()]
by_pat = {p: np.where(pid == p)[0] for p in np.unique(pid)}

def lead_stats(name, s):
    leads, n_ep = [], 0
    for p, ix in by_pat.items():
        tt, sc, th = t_idx[ix], oof[name][ix], thr[name][s][ix]
        for k in episodes[p]:
            m = (tt >= k - 24) & (tt <= k - 6)
            if not m.any():
                continue
            n_ep += 1
            al = m & (sc > th)
            if al.any():
                leads.append(k - tt[al].min())
    L = np.array(leads)
    return dict(episodes=n_ep, detected=float(len(L) / max(n_ep, 1)),
                mean_lead_h=float(L.mean()) if len(L) else None, median_lead_h=float(np.median(L)) if len(L) else None), L

res = {}
for n in ALL:
    r = dict(roc_auc=roc_auc_score(y, oof[n]), pr_auc=average_precision_score(y, oof[n]))
    for s in (90, 95):
        al = oof[n] > thr[n][s]
        r[f"sens{s}"], r[f"spec{s}"] = float(al[y == 1].mean()), float(1 - al[y == 0].mean())
        r[f"false_alerts_per_week{s}"] = float((al & (y == 0)).sum() / (len(df) / 28))
        ls, L = lead_stats(n, s)
        r.update({f"{k}{s}": v for k, v in ls.items()})
        if n == FUSED and s == 90:
            lead_fused = L
    res[n] = r

# ---------------- 3. bootstrap over patients ----------------
keys = list(by_pat)
B = {k: [] for k in ["fused_roc", "fused_pr", "d_pr_ehr", "d_pr_air", "d_pr_vs_logreg", "d_pr_vs_rule"]}
for _ in range(200):
    ii = np.concatenate([by_pat[k] for k in rng.choice(keys, len(keys))])
    yy = y[ii]
    ap = {n: average_precision_score(yy, oof[n][ii]) for n in [FUSED, "Wearables + inhaler + air", "Wearables + inhaler",
                                                              "Logistic regression (all)", "Rule: puffs in last 24h"]}
    B["fused_roc"].append(roc_auc_score(yy, oof[FUSED][ii])); B["fused_pr"].append(ap[FUSED])
    B["d_pr_ehr"].append(ap[FUSED] - ap["Wearables + inhaler + air"])
    B["d_pr_air"].append(ap["Wearables + inhaler + air"] - ap["Wearables + inhaler"])
    B["d_pr_vs_logreg"].append(ap[FUSED] - ap["Logistic regression (all)"])
    B["d_pr_vs_rule"].append(ap[FUSED] - ap["Rule: puffs in last 24h"])
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in B.items()}

# ---------------- 4. extras: early-monitoring subset and calibration ----------------
early = t_idx < 168                                              # first week of monitoring
res["_early_week_pr_auc"] = {n: float(average_precision_score(y[early], oof[n][early])) for n in ALL}
res["_early_week_positives"] = int(y[early].sum())
pf = oof[FUSED]
res["_brier"] = dict(model=float(brier_score_loss(y, pf)), base_rate_only=float(brier_score_loss(y, np.full(len(y), y.mean()))))
json.dump(dict(metrics=res, bootstrap_95ci=ci, n_rows=int(len(df)), n_patients=int(len(keys)), base_rate=float(y.mean())),
          open("results/metrics.json", "w"), indent=1)

# ---------------- 5. demo patients + model config ----------------
ep_count = {p: len(episodes[p]) for p in demo_pool}
ranked = sorted([p for p in demo_pool if ep_count[p] > 0], key=lambda p: ep_count[p])
cfg["demo_patients"] = [int(ranked[i]) for i in np.linspace(0, len(ranked) - 1, 8).round().astype(int)]
json.dump(cfg, open("models/config.json", "w"), indent=1)

# ---------------- 6. figures ----------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
for n, c in zip(["Rule: puffs in last 24h", "EHR only", "Wearables + inhaler", "Wearables + inhaler + air", FUSED],
                ["#999", "#bbb", "#4c78a8", "#2f9e44", "#d9480f"]):
    pr, rc, _ = precision_recall_curve(y, oof[n]); ax[0, 0].plot(rc, pr, color=c, label=f"{n} ({res[n]['pr_auc']:.2f})")
ax[0, 0].axhline(y.mean(), color="k", ls=":", lw=.8); ax[0, 0].set(title="Precision-recall (5-fold, by patient)", xlabel="Recall", ylabel="Precision")
ax[0, 0].legend(fontsize=7, frameon=False)
q = pd.qcut(pf, 10, duplicates="drop"); cal = pd.DataFrame(dict(p=pf, y=y)).groupby(q, observed=True).mean()
ax[0, 1].plot(cal.p, cal.y, "o-", color="#d9480f"); ax[0, 1].plot([0, cal.p.max()], [0, cal.p.max()], "k:")
ax[0, 1].set(title="Calibration (fused)", xlabel="Predicted risk", ylabel="Observed flare rate")
ax[1, 0].hist(lead_fused, bins=[6, 9, 12, 15, 18, 21, 24.5], color="#d9480f", rwidth=.85)
ax[1, 0].set(title="Warning lead time, fused @ 90% specificity", xlabel="Hours before flare", ylabel="Episodes warned")
bst = lgb.Booster(model_file="models/fused_seed0.txt")
samp = df[df.patient_id.isin(demo_pool)].sample(4000, random_state=0)
contrib = bst.predict(samp[cfg["features"]], pred_contrib=True)[:, :-1]
imp = pd.Series(np.abs(contrib).mean(0), index=cfg["features"]).sort_values()
ax[1, 1].barh(imp.tail(12).index, imp.tail(12).values, color="#4c78a8"); ax[1, 1].set_title("Top drivers (mean |TreeSHAP|)")
for a in ax.ravel(): a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/summary.png", dpi=140)

# ---------------- 7. print ----------------
tab = pd.DataFrame(res).T.loc[ALL, ["roc_auc", "pr_auc", "sens90", "spec90", "false_alerts_per_week90",
                                    "detected90", "mean_lead_h90"]].astype(float).round(3)
print("\n", tab.to_string())
print("\npatients", len(keys), "| rows", len(df), "| base rate", round(y.mean(), 3), "| episodes scored", res[FUSED]["episodes90"])
print("95% bootstrap CIs:", {k: [round(a, 3) for a in v] for k, v in ci.items()})
print("first-week PR-AUC:", {k: round(v, 3) for k, v in res["_early_week_pr_auc"].items()}, "| positives", res["_early_week_positives"])
print("Brier (model vs base-rate-only):", {k: round(v, 4) for k, v in res["_brier"].items()})
sh = {g: float(np.abs(contrib[:, [cfg['features'].index(f) for f in cols]]).sum() / np.abs(contrib).sum()) for g, cols in cfg["groups"].items()}
print("share of |SHAP| by group:", {k: round(v, 2) for k, v in sh.items()}, "| demo patients:", cfg["demo_patients"])