"""Synthetic EHR for adults with persistent asthma.

Two files are written:
  ehr.csv         what the model and the dashboard may see
  ehr_latent.csv  hidden 'true susceptibility' used by the simulator, never by the model
"""
import numpy as np
import pandas as pd


def make_ehr(n=300, seed=42):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)
    sex = rng.integers(0, 2, n)                                     # 1 = male
    bmi = np.clip(rng.normal(25.5, 4.5, n), 17, 42).round(1)
    u = rng.random(n)
    p_cur = np.where(sex == 1, 0.25, 0.03)                          # smoking is far more common in men in India
    p_for = np.where(sex == 1, 0.15, 0.02)
    smoking = np.where(u < p_cur, 2, np.where(u < p_cur + p_for, 1, 0))  # 0 never, 1 former, 2 current
    allergy = (rng.random(n) < 0.45).astype(int)
    fev1 = np.clip(82 - 6 * (smoking == 2) - 3 * (smoking == 1) - 0.15 * (age - 38)
                   - 4 * allergy + rng.normal(0, 10, n), 40, 110).round(0)
    adherence = np.clip(rng.beta(5, 3, n), 0.05, 1).round(2)         # share of controller doses taken
    purifier = (rng.random(n) < 0.15).astype(int)
    outdoor_h = np.clip(rng.gamma(2.5, 0.8, n), 0.2, 8).round(1)    # hours outdoors per day

    raw = (0.04 * (100 - fev1) + 0.6 * allergy + 0.4 * (smoking == 2) + 0.2 * (smoking == 1)
           - 1.0 * (adherence - 0.6) + 0.35 * rng.normal(0, 1, n))
    sus = (raw - raw.mean()) / raw.std()                            # hidden susceptibility (z-score)
    flares = np.clip(rng.poisson(np.exp(0.3 + 0.5 * sus)), 0, 8)

    ehr = pd.DataFrame(dict(patient_id=np.arange(n), age=age, sex=sex, bmi=bmi, smoking=smoking,
                            allergy=allergy, fev1_pct=fev1, flares_last_year=flares,
                            controller_adherence=adherence, has_purifier=purifier,
                            outdoor_hours_day=outdoor_h))
    latent = pd.DataFrame(dict(patient_id=np.arange(n), susceptibility=sus.round(3)))
    return ehr, latent


if __name__ == "__main__":
    ehr, latent = make_ehr()
    ehr.to_csv("data/raw/ehr.csv", index=False)
    latent.to_csv("data/raw/ehr_latent.csv", index=False)
    print(ehr.describe().round(2).T[["mean", "std", "min", "max"]])
    print("\nsmoking mix:", ehr.smoking.value_counts(normalize=True).round(2).to_dict())
    print("corr(fev1, flares_last_year):", round(ehr.fev1_pct.corr(ehr.flares_last_year), 2))
    print("corr(adherence, flares_last_year):", round(ehr.controller_adherence.corr(ehr.flares_last_year), 2))