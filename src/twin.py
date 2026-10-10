"""The Digital Twin object - risk, explanation, and what-if scenarios."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from simulate_patients import outdoor_profile


class DigitalTwin:
    def __init__(self, model_dir="models"):
        self.cfg = json.load(open(f"{model_dir}/config.json"))
        self.features = self.cfg["features"]
        self.boosters = [lgb.Booster(model_file=f"{model_dir}/fused_seed{s}.txt") for s in range(5)]

    def risk(self, X):
        """Mean risk of a flare in the next 24 h, plus the spread between the 5 ensemble members."""
        P = np.stack([b.predict(X[self.features]) for b in self.boosters])
        return P.mean(0), P.std(0)

    def explain(self, row):
        """TreeSHAP contribution of every feature (log-odds), and the same summed by data stream."""
        c = np.mean([b.predict(row[self.features], pred_contrib=True)[0][:-1] for b in self.boosters], axis=0)
        s = pd.Series(c, index=self.features)
        groups = {g: float(s[[f for f in cols if f in s.index]].sum()) for g, cols in self.cfg["groups"].items()}
        groups["time"] = float(s.get("hour_of_day", 0.0))
        return s, groups

    def scenario(self, X, env, outdoor_scale=1.0, purifier=None, avoid_above=None):
        """Re-compute the planned-exposure features for the next 24 h under a what-if, keep everything else.

        env must contain the forecast column pm25_fcst (see features.add_forecast).
        Supported levers: time outdoors, a purifier, staying indoors when forecast PM2.5 is high.
        """
        X = X.copy()
        t = X.t.values[:, None] + np.arange(1, 25)
        fc, hour = env.pm25_fcst.values[t], env.timestamp.dt.hour.values[t]
        w = outdoor_profile()
        oh = X.outdoor_hours_day.values * outdoor_scale
        pur = X.has_purifier.values if purifier is None else np.full(len(X), purifier)
        p_out = np.minimum(1.0, oh[:, None] * w[hour] / w.sum())
        if avoid_above is not None:
            p_out = np.where(fc > avoid_above, 0.0, p_out)
        exposure = fc * (p_out + (1 - p_out) * np.where(pur == 1, 0.5, 0.6)[:, None])
        X["exp_next24_mean"], X["exp_next24_max"] = exposure.mean(1), exposure.max(1)
        X["outdoor_hours_day"], X["has_purifier"] = oh, pur
        return X