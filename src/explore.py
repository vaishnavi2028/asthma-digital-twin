"""Build the modelling table, prove there is no leakage, and look at the data."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import (build_dataset, env_features, wearable_features, ENV_COLS, WEAR_COLS, EHR_COLS)

obs = pd.read_csv("data/raw/timeseries.csv.gz", parse_dates=["timestamp"])
env = pd.read_csv("data/raw/environment.csv", parse_dates=["timestamp"])
ehr = pd.read_csv("data/raw/ehr.csv")

# ---------- 1. leakage test: change the future, past features must not move ----------
CUT = 1000
o = obs[obs.patient_id == 7].reset_index(drop=True)
o2, env2 = o.copy(), env.copy()
rng = np.random.default_rng(0)
for c in ["hr", "resp_rate", "spo2", "steps", "rescue_puffs", "sleep_disturb"]:
    o2.loc[CUT + 1:, c] = rng.permutation(o2.loc[CUT + 1:, c].values)
for c in ["pm25", "pm10", "temp_c", "rh_pct"]:
    env2.loc[CUT + 1:, c] = rng.permutation(env2.loc[CUT + 1:, c].values)
a = pd.concat([env_features(env), wearable_features(o)], axis=1).iloc[:CUT + 1]
b = pd.concat([env_features(env2), wearable_features(o2)], axis=1).iloc[:CUT + 1]
assert np.allclose(a.fillna(-1).values, b.fillna(-1).values), "LEAKAGE: features depend on the future!"
print("leakage test passed: scrambling everything after hour", CUT, "changed no earlier feature")

# ---------- 2. build the table ----------
df = build_dataset(obs, env, ehr)
df.to_csv("data/processed/model_table.csv.gz", index=False)
print(f"\nrows {len(df):,} | patients {df.patient_id.nunique()} | features "
      f"{len(ENV_COLS) + len(WEAR_COLS) + len(EHR_COLS) + 1}")
print("share not already flaring (kept for modelling):", round(df.eligible.mean(), 3))
d = df[df.eligible].copy()
print("flare-in-next-24h rate among kept rows:", round(d.label.mean(), 3))

# ---------- 3. does the signal exist? ----------
print("\nrate by PM2.5 24h-mean tercile:")
d["pm_tercile"] = pd.qcut(d.pm25_mean_24h, 3, labels=["clean", "mid", "dirty"])
print(d.groupby("pm_tercile", observed=True).label.mean().round(3).to_dict())
print("rate by purifier:", d.groupby("has_purifier").label.mean().round(3).to_dict())
print("rate by prediction hour:", d.groupby("hour_of_day").label.mean().round(3).to_dict())
feats = ENV_COLS + WEAR_COLS + EHR_COLS
cor = d[feats].corrwith(d.label).sort_values(key=abs, ascending=False)
print("\ntop 8 features by |correlation| with the label:\n", cor.head(8).round(2).to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
d.groupby(pd.qcut(d.pm25_mean_24h, 8)).label.mean().plot(kind="bar", ax=ax[0], color="#d9480f")
ax[0].set_title("Flare rate vs PM2.5 (24h mean, octiles)"); ax[0].set_xticklabels([f"{i+1}" for i in range(8)], rotation=0)
d.groupby(pd.cut(d.flares_last_year, [-1, 0, 1, 2, 4, 9])).label.mean().plot(kind="bar", ax=ax[1], color="#4c78a8")
ax[1].set_title("Flare rate vs flares last year"); ax[1].set_xlabel("")
d.groupby(pd.qcut(d.rr_dev_24h, 8)).label.mean().plot(kind="bar", ax=ax[2], color="#2f9e44")
ax[2].set_title("Flare rate vs breathing-rate deviation"); ax[2].set_xticklabels([f"{i+1}" for i in range(8)], rotation=0)
plt.tight_layout(); plt.savefig("results/eda.png", dpi=140)
print("\nsaved data/processed/model_table.csv.gz and results/eda.png")