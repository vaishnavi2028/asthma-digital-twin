"""Do the twin's what-if answers match the simulator's TRUE counterfactuals?

For every held-out patient and every prediction time t we ask: "if the patient changed X starting NOW,
what happens to the chance of a flare in the next 24 h?"  The simulator answers it exactly by re-running
the same patient with the same random luck. We compare that with the twin's answer.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from features import add_forecast, label_series
from simulate_patients import sim_patient
from twin import DigitalTwin

env = add_forecast(pd.read_csv("data/raw/environment.csv", parse_dates=["timestamp"]))
ehr = pd.read_csv("data/raw/ehr.csv")
sus = pd.read_csv("data/raw/ehr_latent.csv").set_index("patient_id").susceptibility
df = pd.read_csv("data/processed/model_table.csv.gz")
df = df[df.eligible].reset_index(drop=True)
twin = DigitalTwin()

# held-out patients = the ones the saved ensemble never trained on (fold 0 of the evaluation)
_, te = next(iter(GroupKFold(5).split(df, groups=df.patient_id)))
test_ids = np.unique(df.patient_id.values[te])
rows = df[df.patient_id.isin(test_ids)].reset_index(drop=True)

# sanity: a scenario that changes nothing must reproduce the stored features
same = twin.scenario(rows, env)
assert np.allclose(same[["exp_next24_mean", "exp_next24_max"]].values, rows[["exp_next24_mean", "exp_next24_max"]].values, rtol=1e-6)
print("scenario engine reproduces stored features when nothing changes: OK")

SCEN = {  # name: (who is targeted, simulator change, twin change)
    "Add an air purifier": (lambda e: e.has_purifier == 0, {"has_purifier": 1}, dict(purifier=1)),
    "Halve time outdoors": (lambda e: True, {"outdoor_scale": 0.5}, dict(outdoor_scale=0.5)),
    "Stay indoors when PM2.5 > 150": (lambda e: True, {"avoid_above": 150}, dict(avoid_above=150)),
    "Controller adherence to 90%": (lambda e: e.controller_adherence < 0.9, {"controller_adherence": 0.9}, None),
}
rng = np.random.default_rng(0)
out = []
for name, (sel, sim_cf, twin_cf) in SCEN.items():
    tg = [e for e in ehr[ehr.patient_id.isin(test_ids)].itertuples() if sel(e)]
    X = rows[rows.patient_id.isin([e.patient_id for e in tg])].reset_index(drop=True)
    p0, _ = twin.risk(X)
    p1 = twin.risk(twin.scenario(X, env, **twin_cf))[0] if twin_cf else p0
    y_cf = np.zeros(len(X))
    for e in tg:                                                  # true answer: re-simulate from each prediction time
        ix = np.where(X.patient_id.values == e.patient_id)[0]
        for i in ix:
            o1, _ = sim_patient(e, sus[e.patient_id], env, 42, cf={"start": int(X.t.values[i]), **sim_cf})
            y_cf[i] = label_series(o1.rescue_puffs.values)[0][int(X.t.values[i])]
    per = pd.DataFrame(dict(yf=X.label.values, yc=y_cf, p0=p0, p1=p1)).groupby(X.patient_id.values).sum()
    bt, bm = [], []
    for _ in range(300):
        b = per.iloc[rng.integers(0, len(per), len(per))]
        bt.append(1 - b.yc.sum() / b.yf.sum()); bm.append(1 - b.p1.sum() / b.p0.sum())
    out.append(dict(scenario=name, patients=len(tg), true_reduction=1 - per.yc.sum() / per.yf.sum(),
                    true_ci=np.percentile(bt, [2.5, 97.5]).round(2), twin_reduction=1 - per.p1.sum() / per.p0.sum(),
                    twin_ci=np.percentile(bm, [2.5, 97.5]).round(2), supported=twin_cf is not None))
    print("done", name, flush=True)
res = pd.DataFrame(out)
res.to_csv("results/whatif_validation.csv", index=False)
print(res.round(2).to_string(index=False))