"""Hourly wearable + smart-inhaler simulator with a hidden airway-inflammation state.

Story the simulator tells (this is what your Digital Twin has to rediscover):
  city air + weather + where the patient spends time  ->  inhaled irritant dose
  dose accumulates in a slowly-decaying hidden 'inflammation' state (half-life ~2 days)
  hidden state + random triggers (viral illness, shocks)  ->  symptom score S
  S  ->  higher heart/respiratory rate, lower SpO2, worse sleep, fewer steps, more rescue puffs
The model never sees S or the inflammation state; it only sees the wearable columns, EHR and air data.
"""
import sys
import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.insert(0, "src")
from environment import ar1  # noqa: E402

DECAY = 0.985          # inflammation memory per hour (half-life about 46 h)


def outdoor_profile():
    """Relative chance of being outdoors by hour: commute peaks plus a lunchtime bump."""
    h = np.arange(24)
    w = (0.1 + np.exp(-((h - 8.5) / 1.2) ** 2) + np.exp(-((h - 18) / 1.5) ** 2)
         + 0.5 * np.exp(-((h - 13) / 2) ** 2))
    w[(h < 6) | (h > 21)] = 0
    return w


def sim_patient(e, sus, env, seed=42):
    """Simulate one patient. Seeded per patient so counterfactuals (e.g. add a purifier) reuse the same luck."""
    rng = np.random.default_rng([seed, int(e.patient_id)])
    n = len(env)
    hour = env.timestamp.dt.hour.values
    pm, temp = env.pm25.values, env.temp_c.values

    # ---- daily routine ----
    bed, wake = int(np.clip(round(rng.normal(23, 0.8)), 22, 23)), int(np.clip(round(rng.normal(6.5, 0.6)), 6, 7))
    asleep = (hour >= bed) | (hour < wake)
    w = outdoor_profile()
    p_out = np.minimum(1.0, e.outdoor_hours_day * w / w.sum())
    outdoors = rng.random(n) < p_out[hour]

    # ---- what the patient actually breathes ----
    infil = 0.5 if e.has_purifier else 0.6                    # indoor share of outdoor PM2.5
    exposure = pm * np.where(outdoors, 1.0, infil)
    excess = np.maximum(exposure - 50, 0) / 100                # only exposure above ~50 ug/m3 irritates
    cold = np.maximum(12 - temp, 0) / 10 * outdoors            # cold air is an extra trigger outdoors
    drive = excess + 0.3 * cold

    # ---- hidden airway state ----
    sens = np.exp(0.45 * np.clip(sus, -2, 2) + 0.15 * rng.standard_normal()) * (1 - 0.5 * e.controller_adherence)
    infl = lfilter([1], [1, -DECAY], sens * 0.05 * drive)      # slow accumulation of exposure
    acute = 0.15 * sens * excess                               # immediate reaction

    infection = np.zeros(n)                                    # viral illness: unobserved, unpredictable
    for _ in range(rng.poisson(1.2)):
        s, d = rng.integers(0, n - 200), rng.integers(96, 192)
        infection[s:s + d] += rng.uniform(0.6, 1.2) * np.hanning(d)
    shocks = lfilter([1], [1, -0.9], (rng.random(n) < 1 / 120) * rng.exponential(0.8, n))
    S = infl + acute + infection + shocks                      # symptom score

    # ---- rescue inhaler (smart inhaler) ----
    lam = 0.02 + 0.45 * np.maximum(S - 1.8, 0) ** 1.2
    lam = lam * np.where(asleep, 1.3, 1.0)                     # asthma is worse at night
    puffs = rng.poisson(lam)

    # ---- wearable signals ----
    daily_steps = rng.uniform(3000, 9000)
    shape = np.where(asleep, 0, 0.3 + w[hour] / w.max())
    steps = rng.poisson(daily_steps * shape / (shape.reshape(-1, 24).sum(1).mean()) * np.exp(-0.4 * S))
    base_hr = 68 + 0.1 * (e.age - 38) + rng.normal(0, 4)
    hr = (base_hr - 9 * asleep + 0.015 * np.minimum(steps, 1800) + 5 * S
          + ar1(n, 0.8, 1.0, rng) + rng.normal(0, 2, n))
    rr = np.clip(15 - 1.5 * asleep + 0.0008 * steps + 2.2 * S + ar1(n, 0.7, 0.4, rng) + rng.normal(0, 0.8, n), 8, 40)
    spo2 = np.clip(98.2 - 0.03 * (100 - e.fev1_pct) - 0.4 * asleep - 1.0 * S + rng.normal(0, 0.6, n), 80, 100)
    disturb = rng.poisson(np.where(asleep, 0.1 + 0.6 * S, 0))

    obs = pd.DataFrame(dict(patient_id=int(e.patient_id), timestamp=env.timestamp.values,
                            hr=hr.round(0), resp_rate=rr.round(1), spo2=spo2.round(0), steps=steps,
                            asleep=asleep.astype(int), sleep_disturb=disturb, rescue_puffs=puffs))
    hidden = pd.DataFrame(dict(patient_id=int(e.patient_id), timestamp=env.timestamp.values,
                               symptom_score=S.round(3), outdoors=outdoors.astype(int)))
    return obs, hidden


def flare_hours(puffs):
    """SPEC definition: at least 4 puffs inside any 6-hour window."""
    return pd.Series(puffs).rolling(6, min_periods=1).sum().values >= 4


def main(seed=42):
    env = pd.read_csv("data/raw/environment.csv", parse_dates=["timestamp"])
    ehr = pd.read_csv("data/raw/ehr.csv")
    sus = pd.read_csv("data/raw/ehr_latent.csv").set_index("patient_id").susceptibility
    obs, hid = zip(*[sim_patient(e, sus[e.patient_id], env, seed) for e in ehr.itertuples()])
    obs, hid = pd.concat(obs), pd.concat(hid)
    obs.to_csv("data/raw/timeseries.csv.gz", index=False)
    hid.to_csv("data/raw/hidden_state.csv.gz", index=False)   # for plots/checks only, never for modelling
    return ehr, sus, obs, hid


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ehr, sus, obs, hid = main()
    print(f"wrote {obs.patient_id.nunique()} patients x {len(obs) // obs.patient_id.nunique()} hours")
    fl = obs.groupby("patient_id").rescue_puffs.apply(lambda s: flare_hours(s.values))
    per_pat = pd.DataFrame({"flare_hours": fl.apply(lambda a: a.sum()),
                            "episodes": fl.apply(lambda a: int(((a[1:] & ~a[:-1]).sum()) + a[0]))})
    per_pat = per_pat.join(sus).join(ehr.set_index("patient_id")[["flares_last_year"]])
    print("\nflare episodes per patient (90 days):", per_pat.episodes.describe().round(1).to_dict())
    print("patients with zero flares:", round((per_pat.episodes == 0).mean(), 2))
    print("corr(susceptibility, episodes):", round(per_pat.susceptibility.corr(per_pat.episodes), 2))
    print("corr(flares_last_year, episodes):", round(per_pat.flares_last_year.corr(per_pat.episodes), 2))
    print("\nmean signals, calm vs symptomatic hours:")
    m = obs.merge(hid, on=["patient_id", "timestamp"])
    m["state"] = np.where(m.symptom_score > 1.5, "symptomatic", "calm")
    print(m.groupby("state")[["hr", "resp_rate", "spo2", "steps", "sleep_disturb", "rescue_puffs"]].mean().round(2))

    pid = [per_pat.susceptibility.idxmax(), per_pat.susceptibility.idxmin()]
    env = pd.read_csv("data/raw/environment.csv", parse_dates=["timestamp"])
    fig, ax = plt.subplots(4, 2, figsize=(13, 8), sharex="col")
    sl = slice(24 * 40, 24 * 55)
    for j, p in enumerate(pid):
        o, h = obs[obs.patient_id == p].iloc[sl], hid[hid.patient_id == p].iloc[sl]
        ax[0, j].plot(env.timestamp.iloc[sl], env.pm25.iloc[sl], color="#999"); ax[0, j].set_title(f"patient {p}  (susceptibility {sus[p]:+.1f})")
        ax[1, j].plot(o.timestamp, o.resp_rate, color="#4c78a8")
        ax[2, j].plot(o.timestamp, o.spo2, color="#2f9e44")
        ax[3, j].bar(o.timestamp, o.rescue_puffs, width=0.04, color="#d9480f")
    for a, t in zip(ax[:, 0], ["PM2.5", "resp rate", "SpO2 %", "rescue puffs"]): a.set_ylabel(t)
    plt.tight_layout(); plt.savefig("results/patients.png", dpi=140)
    print("\nsaved results/patients.png")