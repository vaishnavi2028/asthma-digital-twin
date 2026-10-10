"""Leak-free features and labels.

Golden rule: a feature at prediction time t may use data from hours <= t and nothing later.
Label: will a NEW flare (>= 4 rescue puffs inside a 6-hour window) start in the next 24 hours?
We predict every 6 hours, only when the patient is not already flaring.
"""
import numpy as np
import pandas as pd

HORIZON, STEP, START = 24, 6, 72     # look 24 h ahead, predict every 6 h, need 72 h of history

EHR_COLS = ["age", "sex", "bmi", "smoking", "allergy", "fev1_pct", "flares_last_year",
            "controller_adherence", "has_purifier", "outdoor_hours_day"]
ENV_COLS = ["pm25_now", "pm25_mean_6h", "pm25_mean_24h", "pm25_mean_72h", "pm25_max_24h",
            "pm25_hours_above_120_24h", "pm25_trend_24h", "pm10_mean_24h", "temp_now", "temp_min_24h", "rh_mean_24h"]
WEAR_COLS = ["hr_mean_6h", "hr_mean_24h", "hr_mean_72h", "rr_mean_6h", "rr_mean_24h", "rr_mean_72h",
             "spo2_mean_6h", "spo2_mean_24h", "spo2_mean_72h", "spo2_min_6h", "hr_dev_24h", "rr_dev_24h",
             "spo2_dev_24h", "rr_trend_24h", "steps_6h", "steps_24h", "steps_ratio_24h", "disturb_24h",
             "puffs_6h", "puffs_24h", "puffs_72h", "hours_since_puff"]
TIME_COLS = ["hour_of_day"]


def env_features(env):
    pm = env.pm25
    f = pd.DataFrame(index=env.index)
    f["pm25_now"] = pm
    for w in (6, 24, 72):
        f[f"pm25_mean_{w}h"] = pm.rolling(w).mean()
    f["pm25_max_24h"] = pm.rolling(24).max()
    f["pm25_hours_above_120_24h"] = (pm > 120).rolling(24).sum()
    f["pm25_trend_24h"] = f.pm25_mean_24h - pm.shift(24).rolling(24).mean()
    f["pm10_mean_24h"] = env.pm10.rolling(24).mean()
    f["temp_now"], f["temp_min_24h"] = env.temp_c, env.temp_c.rolling(24).min()
    f["rh_mean_24h"] = env.rh_pct.rolling(24).mean()
    return f


def wearable_features(o):
    f = pd.DataFrame(index=o.index)
    for col, nm in [("hr", "hr"), ("resp_rate", "rr"), ("spo2", "spo2")]:
        s = o[col].astype(float)
        for w in (6, 24, 72):
            f[f"{nm}_mean_{w}h"] = s.rolling(w).mean()
        # deviation from the patient's OWN past average (expanding = only hours up to now)
        if nm != "spo2":
            f[f"{nm}_dev_24h"] = f[f"{nm}_mean_24h"] - s.expanding(min_periods=72).mean()
    f["spo2_dev_24h"] = f.spo2_mean_24h - o.spo2.astype(float).expanding(min_periods=72).mean()
    f["spo2_min_6h"] = o.spo2.astype(float).rolling(6).min()
    f["rr_trend_24h"] = f.rr_mean_24h - o.resp_rate.shift(24).rolling(24).mean()
    f["steps_6h"], f["steps_24h"] = o.steps.rolling(6).sum(), o.steps.rolling(24).sum()
    f["steps_ratio_24h"] = f.steps_24h / (o.steps.expanding(min_periods=72).mean() * 24 + 1e-6)
    f["disturb_24h"] = o.sleep_disturb.rolling(24).sum()
    p = o.rescue_puffs
    for w in (6, 24, 72):
        f[f"puffs_{w}h"] = p.rolling(w).sum()
    last = pd.Series(np.where(p.values > 0, np.arange(len(p)), np.nan)).ffill().values
    f["hours_since_puff"] = np.clip(np.arange(len(p)) - np.nan_to_num(last, nan=-999), 0, 72)
    return f


def label_series(puffs):
    """1 if a flare window lies entirely in the future (hours t+1 .. t+24), else 0."""
    roll6 = pd.Series(puffs).rolling(6).sum().values
    flare_window = pd.Series((roll6 >= 4).astype(float))        # window ending at k covers k-5..k
    fut = flare_window[::-1].rolling(HORIZON - 5).max()[::-1]    # max over windows ending t+6 .. t+24
    return fut.shift(-6).values, roll6


def build_patient(o, env_f):
    n = len(o)
    idx = np.arange(START, n - HORIZON, STEP)
    wf = wearable_features(o.reset_index(drop=True))
    label, roll6 = label_series(o.rescue_puffs.values)
    df = pd.concat([env_f.iloc[idx].reset_index(drop=True), wf.iloc[idx].reset_index(drop=True)], axis=1)
    df.insert(0, "t", idx)
    df.insert(0, "patient_id", o.patient_id.iloc[0])
    df["hour_of_day"] = pd.to_datetime(o.timestamp.values[idx]).hour
    df["label"] = label[idx]
    df["eligible"] = roll6[idx] < 4                                # not already mid-flare
    return df


def build_dataset(obs, env, ehr):
    env_f = env_features(env)
    parts = [build_patient(g.reset_index(drop=True), env_f) for _, g in obs.groupby("patient_id")]
    df = pd.concat(parts, ignore_index=True).merge(ehr, on="patient_id")
    return df.dropna().reset_index(drop=True)