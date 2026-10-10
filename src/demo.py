"""One patient, one moment: risk, why, and what-if."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import add_forecast
from twin import DigitalTwin

env = add_forecast(pd.read_csv("data/raw/environment.csv", parse_dates=["timestamp"]))
df = pd.read_csv("data/processed/model_table.csv.gz")
df = df[df.eligible].reset_index(drop=True)
twin = DigitalTwin()

pid = twin.cfg["demo_patients"][-1]
rows = df[df.patient_id == pid].reset_index(drop=True)
p, spread = twin.risk(rows)
# pick a moment that is elevated but not already extreme, with heavy air pollution ahead: the interesting case
cand = np.where((p > 0.10) & (p < 0.50))[0]
i = int(cand[np.argmax(rows.exp_next24_mean.values[cand])]) if len(cand) else int(np.argmax(p))
row = rows.iloc[[i]]
print(f"Patient {pid} | purifier={int(row.has_purifier.iloc[0])} | outdoor hours/day={row.outdoor_hours_day.iloc[0]}")
print(f"Risk of a flare in the next 24 h: {p[i]:.0%}  (ensemble spread +/- {spread[i]:.1%})  | alert threshold {twin.cfg['thr90']:.0%}")
print("Did a flare actually follow?", bool(row.label.iloc[0]))

contribs, groups = twin.explain(row)
print("\nContribution by data stream (log-odds, + raises risk):", {k: round(v, 2) for k, v in groups.items()})
top = contribs.reindex(contribs.abs().sort_values(ascending=False).index)[:6]
print("\nTop drivers:")
for f, c in top.items():
    print(f"  {f:22s} value {row[f].iloc[0]:8.2f}   contribution {c:+.2f}")

print("\nWhat-if (change starts now):")
whatifs = {"Add an air purifier": dict(purifier=1), "Halve time outdoors": dict(outdoor_scale=0.5),
           "Stay indoors when PM2.5 > 150": dict(avoid_above=150)}
res = {"Today's plan": p[i]}
for name, kw in whatifs.items():
    res[name] = twin.risk(twin.scenario(row, env, **kw))[0][0]
    print(f"  {name:32s} {p[i]:.1%} -> {res[name]:.1%}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
top[::-1].plot(kind="barh", ax=ax[0], color=["#d9480f" if v > 0 else "#2f9e44" for v in top[::-1]])
ax[0].set_title("Why this risk? (TreeSHAP)")
pd.Series(res).plot(kind="barh", ax=ax[1], color="#4c78a8"); ax[1].set_title("Risk under each plan (conservative estimates)")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/demo.png", dpi=140)