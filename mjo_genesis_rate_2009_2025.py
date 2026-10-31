"""
MJO phase and North Atlantic tropical storm genesis RATE, 2009-2025 (June-November).

Builds on mjo_lead_lag_2009_2025.py (same data, same definitions) but replaces raw
counts with a genesis rate that accounts for how often each MJO phase occurs:

    rate(phase, lag) = storms whose MJO state `lag` days before genesis was in `phase`
                       ---------------------------------------------------------------
                       number of active-MJO days in `phase` during Jun-Nov 2009-2025

Rates are shown relative to the all-phase average (1.0 = average). The headline
statistic is the ratio of the genesis rate in Wheeler-Hendon phases 1-2 to that in
phases 6-7, at each lag from 0 to 21 days. A 95% confidence band comes from a
bootstrap that resamples whole hurricane seasons (years) with replacement.

Usage
  python3 mjo_genesis_rate_2009_2025.py HURDAT2_FILE [MJO_FILE] [OUTPUT_PNG]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from mjo_lead_lag_2009_2025 import read_hurdat2, wh_phase

YEARS = np.arange(2009, 2026)
SEASON = range(6, 12)          # June-November
LAGS = np.arange(0, 22)
AMP_MIN = 1.0
PHASES = np.arange(1, 9)
N_BOOT = 5000
SHOW_LAGS = (0, 12)

NAVY, BLUE, GREEN, MUTED, LINE = "#0b2a4a", "#1f5fa8", "#1f8a5b", "#5b6b7a", "#d9e2ea"


def tables(gen, mjo):
    """Per-year arrays: storms[lag, year, phase] and days[year, phase]."""
    yi = {y: i for i, y in enumerate(YEARS)}
    storms = np.zeros((len(LAGS), len(YEARS), 8))
    for k, lag in enumerate(LAGS):
        d = mjo.reindex(gen.date - pd.Timedelta(days=int(lag)))
        ok = (d.amp >= AMP_MIN).values
        for y, ph in zip(gen.time.dt.year.values[ok], d.phase.values[ok]):
            storms[k, yi[y], int(ph) - 1] += 1
    days = np.zeros((len(YEARS), 8))
    s = mjo[(mjo.index.year.isin(YEARS)) & (mjo.index.month.isin(SEASON)) & (mjo.amp >= AMP_MIN)]
    for y, ph in zip(s.index.year, s.phase):
        days[yi[y], int(ph) - 1] += 1
    return storms, days


def ratio(st, dy):
    r12 = st[..., 0:2].sum(-1) / dy[..., 0:2].sum(-1)
    r67 = st[..., 5:7].sum(-1) / dy[..., 5:7].sum(-1)
    return r12 / r67


def main():
    hurdat = sys.argv[1]
    mjo_file = sys.argv[2] if len(sys.argv) > 2 else "data_MJO.txt"
    out_png = sys.argv[3] if len(sys.argv) > 3 else "mjo_genesis_rate_2009_2025.png"

    bt = read_hurdat2(hurdat)
    ts = bt[bt.status.isin(["TS", "HU", "SS"]) & (bt.vmax >= 34)]
    gen = ts.sort_values("time").groupby("id").first().reset_index()
    gen = gen[gen.time.dt.year.isin(YEARS) & gen.time.dt.month.isin(SEASON)].copy()
    gen["date"] = gen.time.dt.normalize()

    mjo = pd.read_csv(mjo_file, sep=r"\s+", header=None,
                      names=["year", "month", "day", "flag", "RMM1", "RMM2", "amp"])
    mjo.index = pd.to_datetime(mjo[["year", "month", "day"]])
    mjo["phase"] = wh_phase(mjo.RMM1.values, mjo.RMM2.values)

    storms, days = tables(gen, mjo)            # (lag, year, phase), (year, phase)
    S, D = storms.sum(1), days.sum(0)          # totals over years
    rate = S / D
    rel = rate / rate.mean(1, keepdims=True)
    obs = ratio(S, D)

    rng = np.random.default_rng(42)
    boots = np.empty((N_BOOT, len(LAGS)))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(YEARS), len(YEARS))
        boots[b] = ratio(storms[:, idx].sum(1), days[idx].sum(0))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    frac_gt1 = (boots > 1).mean(0)

    peak = int(np.argmax(obs))
    print(f"Jun-Nov TS genesis events {YEARS[0]}-{YEARS[-1]}: {len(gen)}")
    print("Active Jun-Nov days per WH phase:", D.astype(int).tolist())
    out = pd.DataFrame({"lag": LAGS, "n_storms": S.sum(1).astype(int), "ratio_P12_P67": obs,
                        "ci_lo": lo, "ci_hi": hi, "boot_frac_gt1": frac_gt1})
    print(out.round(2).to_string(index=False))
    for lag in SHOW_LAGS:
        print(f"Lag {lag} relative rate by phase:", np.round(rel[lag], 2).tolist(),
              "| storms:", S[lag].astype(int).tolist())
    print(f"Peak ratio at lag {peak}: {obs[peak]:.2f} (95% CI {lo[peak]:.2f}-{hi[peak]:.2f})")
    out.to_csv(out_png.replace(".png", "_stats.csv"), index=False)

    # ---------------- figure ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "axes.labelcolor": NAVY, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.1, 1]})

    w = 0.38
    a1.axhline(1, color=MUTED, lw=1, ls="--", zorder=1)
    a1.bar(PHASES - w / 2 - 0.01, rel[SHOW_LAGS[0]], w, color=BLUE, zorder=2,
           label=f"MJO phase on genesis day (lag {SHOW_LAGS[0]})")
    a1.bar(PHASES + w / 2 + 0.01, rel[SHOW_LAGS[1]], w, color=GREEN, hatch="//",
           edgecolor="white", linewidth=0, zorder=2,
           label=f"MJO phase {SHOW_LAGS[1]} days before genesis")
    for x0, x1 in [(0.5, 2.5)]:
        a1.axvspan(x0, x1, color=GREEN, alpha=0.06, zorder=0)
    a1.axvspan(5.5, 7.5, color=MUTED, alpha=0.08, zorder=0)
    a1.text(1.5, 2.12, "Phases 1–2\nAfrica / W. Indian Ocean", ha="center", va="top",
            fontsize=10, color=NAVY)
    a1.text(6.5, 2.12, "Phases 6–7\nWestern Pacific", ha="center", va="top",
            fontsize=10, color=NAVY)
    a1.set_ylim(0, 2.2)
    a1.set_xticks(PHASES)
    a1.set_xlabel("MJO phase (Wheeler–Hendon)", fontsize=12)
    a1.set_ylabel("Genesis rate relative to average (1 = average)", fontsize=12)
    a1.set_title("a) Storm formation rate by MJO phase", fontsize=13, color=NAVY,
                 loc="left", fontweight="bold")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10)

    a2.fill_between(LAGS, lo, hi, color=GREEN, alpha=0.15, lw=0, label="95% bootstrap range")
    a2.axhline(1, color=MUTED, lw=1, ls="--")
    a2.plot(LAGS, obs, color=GREEN, lw=2.5, marker="o", ms=7, mec="white", mew=1.5,
            label="Observed ratio")
    a2.annotate(f"Peak: {obs[peak]:.1f}× at {LAGS[peak]} days",
                xy=(LAGS[peak], obs[peak]), xytext=(LAGS[peak] + 2.5, obs[peak] + 0.9),
                fontsize=11, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=1))
    a2.text(21.3, 0.35, "dashed line = no difference", ha="right", va="center", fontsize=9, color=MUTED)
    a2.set_xlim(-0.5, 21.5)
    a2.set_xticks(np.arange(0, 22, 3))
    a2.set_ylim(0, max(4.5, np.nanmax(hi) * 1.05))
    a2.set_xlabel("Lag: days between MJO phase and storm formation", fontsize=12)
    a2.set_ylabel("Rate in phases 1–2 ÷ rate in phases 6–7", fontsize=12)
    a2.set_title("b) How much more often storms form after phases 1–2", fontsize=13,
                 color=NAVY, loc="left", fontweight="bold")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10)

    for ax in (a1, a2):
        ax.grid(axis="y", color=LINE, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(f"North Atlantic tropical storm formation and the MJO, June–November "
                 f"{YEARS[0]}–{YEARS[-1]} (n = {len(gen)} storms, active MJO days only)",
                 fontsize=14, color=NAVY, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, facecolor="white")
    print("Saved", out_png)


if __name__ == "__main__":
    main()
