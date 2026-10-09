"""Hourly Delhi-like winter environment stream (PM2.5, PM10, temperature, humidity).

Why simulate? Real Delhi archives exist (CPCB, OpenAQ) but are hard to download in bulk and
have gaps. This generator gives a reproducible stream with realistic structure; Ends with
a hook to swap in real PM2.5 later.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter


def ar1(n, phi, sd, rng):
    """Autocorrelated noise: today's value remembers yesterday's."""
    return lfilter([1], [1, -phi], rng.normal(0, sd, n))


def simulate_environment(start="2025-11-01", days=90, seed=42):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 24, freq="h")
    n = len(t)
    hour = t.hour.values
    day = np.arange(n) / 24

    # ---- weather first (it influences pollution) ----
    temp = 23 - 0.11 * day + 6 * np.cos(2 * np.pi * (hour - 15) / 24) + ar1(n, 0.99, 0.25, rng)
    rh = np.clip(70 - 1.8 * (temp - 18) + ar1(n, 0.98, 1.0, rng), 15, 100)

    # ---- PM2.5 in log space: baseline + daily rhythm + episodes + weather + noise ----
    diurnal = (0.25 * np.exp(-((hour - 8) / 2.0) ** 2)        # morning traffic peak
               + 0.30 * np.exp(-((hour - 21.5) / 2.5) ** 2)    # evening peak, low boundary layer
               - 0.35 * np.exp(-((hour - 15) / 3.5) ** 2))     # afternoon dip
    episodes = np.zeros(n)
    for _ in range(rng.integers(3, 6)):                        # multi-day smog episodes
        centre = rng.uniform(5, days - 5) * 24
        width = rng.uniform(1.0, 2.5) * 24
        episodes += rng.uniform(0.5, 0.9) * np.exp(-((np.arange(n) - centre) / width) ** 2)
    stubble_tail = 0.5 * np.exp(-day / 12)                     # post-harvest burning fades out
    log_pm = (np.log(95) + diurnal + episodes + stubble_tail
              - 0.025 * (temp - 18) + ar1(n, 0.97, 0.08, rng))
    pm25 = np.clip(np.exp(log_pm), 8, 800)
    pm10 = pm25 * np.clip(1.7 + ar1(n, 0.9, 0.05, rng), 1.2, 2.5)

    # Band labels follow India's CPCB PM2.5 breakpoints (verify against the current CPCB table)
    bins = [0, 30, 60, 90, 120, 250, np.inf]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]
    return pd.DataFrame({
        "timestamp": t, "pm25": pm25.round(1), "pm10": pm10.round(1),
        "temp_c": temp.round(1), "rh_pct": rh.round(0),
        "pm25_band": pd.cut(pm25, bins=bins, labels=labels),
    })


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = simulate_environment()
    env.to_csv("data/raw/environment.csv", index=False)
    print(env.describe().round(1))
    print(env.pm25_band.value_counts(normalize=True).round(2))

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    daily = env.set_index("timestamp").pm25.resample("D").mean()
    ax[0].plot(daily.index, daily.values, color="#d9480f")
    ax[0].axhline(60, color="grey", ls="--", lw=0.8)
    ax[0].set_title("Daily mean PM2.5 (ug/m3), 90 days")
    wk = env.iloc[24 * 20:24 * 27]
    ax[1].plot(wk.timestamp, wk.pm25, color="#4c78a8")
    ax[1].set_title("One week, hourly: daily rhythm")
    plt.tight_layout()
    plt.savefig("results/environment.png", dpi=150)
    print("saved data/raw/environment.csv and results/environment.png")