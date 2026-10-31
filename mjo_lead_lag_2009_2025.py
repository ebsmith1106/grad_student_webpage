"""
MJO lead-lag relationship with North Atlantic tropical storm genesis, 2009-2025.

Inputs
  hurdat2 file  : NHC Atlantic best track (HURDAT2), e.g. hurdat2-1851-2025-091226.txt
  data_MJO.txt  : daily RMM index (year month day 0 RMM1 RMM2 amplitude)

Method
  * Tropical storm genesis = first best-track fix at which a system is a tropical
    or subtropical cyclone (status TS, HU or SS) with winds >= 34 kt.
    Storms are identified by their unique HURDAT2 ID (e.g. AL012020), not by name.
  * MJO phase uses the standard Wheeler & Hendon (2004) convention:
        phase 1 = 180-225 deg in (RMM1, RMM2) space, increasing counterclockwise.
  * For each lag L (0-21 days), each genesis is paired with the MJO state L days
    before genesis. Only days with RMM amplitude >= 1 (an active MJO) are kept.
  * Genesis counts per phase are fit with a quadratic trendline (R^2 reported),
    and the Pearson r between phase number and count is reported, matching the
    original class-project figure.

Usage
  python3 mjo_lead_lag_2009_2025.py HURDAT2_FILE [MJO_FILE] [OUTPUT_PNG]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chisquare

YEAR_START, YEAR_END = 2009, 2025
LAGS = np.arange(0, 22)
AMP_MIN = 1.0
PHASES = np.arange(1, 9)


def read_hurdat2(path):
    rows, sid = [], None
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if parts[0].startswith("AL") and len(parts[0]) == 8:
                sid, name = parts[0], parts[1]
                continue
            t = pd.Timestamp(f"{parts[0]} {parts[1][:2]}:{parts[1][2:]}")
            rows.append((sid, name, t, parts[3], int(parts[6])))
    return pd.DataFrame(rows, columns=["id", "name", "time", "status", "vmax"])


def wh_phase(rmm1, rmm2):
    """Wheeler-Hendon phase 1-8 (phase 1 starts at 180 deg)."""
    ang = (np.degrees(np.arctan2(rmm2, rmm1)) + 360) % 360
    return (((ang - 180) % 360) // 45 + 1).astype(int)


def poly_r2(x, y, order=2):
    p = np.poly1d(np.polyfit(x, y, order))
    ss_res = np.sum((y - p(x)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, p


def main():
    hurdat = sys.argv[1]
    mjo_file = sys.argv[2] if len(sys.argv) > 2 else "data_MJO.txt"
    out_png = sys.argv[3] if len(sys.argv) > 3 else "lead_lag_relationship_2009_2025.png"

    bt = read_hurdat2(hurdat)
    ts = bt[bt.status.isin(["TS", "HU", "SS"]) & (bt.vmax >= 34)]
    gen = ts.sort_values("time").groupby("id").first().reset_index()
    gen = gen[(gen.time.dt.year >= YEAR_START) & (gen.time.dt.year <= YEAR_END)].copy()
    gen["date"] = gen.time.dt.normalize()
    print(f"Tropical storm genesis events {YEAR_START}-{YEAR_END}: {len(gen)} "
          f"(unique names: {gen.name.nunique()})")

    mjo = pd.read_csv(mjo_file, sep=r"\s+", header=None,
                      names=["year", "month", "day", "flag", "RMM1", "RMM2", "amp"])
    mjo["date"] = pd.to_datetime(mjo[["year", "month", "day"]])
    mjo["phase"] = wh_phase(mjo.RMM1.values, mjo.RMM2.values)
    mjo = mjo.set_index("date")

    results = []
    for lag in LAGS:
        d = mjo.reindex(gen.date - pd.Timedelta(days=int(lag)))
        d = d[d.amp >= AMP_MIN]
        counts = d.phase.value_counts().reindex(PHASES, fill_value=0).values.astype(float)
        r, p_r = pearsonr(PHASES, counts)
        r2, _ = poly_r2(PHASES, counts)
        chi_p = chisquare(counts).pvalue
        results.append(dict(lag=lag, n=int(counts.sum()), r=r, p_r=p_r, r2=r2,
                            chi_p=chi_p, counts=counts))
    res = pd.DataFrame(results)
    peak = res.loc[res.r2.idxmax()]
    print(res[["lag", "n", "r", "p_r", "r2", "chi_p"]].round(3).to_string(index=False))
    print(f"Peak R^2 at lag {int(peak.lag)}: R^2={peak.r2:.3f}, r={peak.r:+.3f}")
    for lag in [0, int(peak.lag)]:
        print(f"Lag {lag} counts by WH phase:", res.loc[lag, "counts"].astype(int).tolist())

    # ---------------- figure ----------------
    mid = 12 if int(peak.lag) != 12 else 7
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panels = [(axes[0, 0], 0, "steelblue", "red", f"Lag 0 Days (Immediate)"),
              (axes[0, 1], mid, "orange", "red", f"Lag {mid} Days"),
              (axes[1, 0], int(peak.lag), "brown", "darkgreen",
               f"Lag {int(peak.lag)} Days (Peak Signal) ★")]
    for ax, lag, bar_c, line_c, title in panels:
        row = res.loc[lag]
        ax.bar(PHASES, row.counts, color=bar_c, alpha=0.6, edgecolor="black",
               linewidth=1.5, label="Counts")
        _, p = poly_r2(PHASES, row.counts)
        xs = np.linspace(1, 8, 200)
        ax.plot(xs, p(xs), color=line_c, lw=2.5, label="Trendline (order 2)")
        tcolor = "darkgreen" if lag == int(peak.lag) else "black"
        ax.set_title(f"{title}\n($R^2$ = {row.r2:.3f}, r = {row.r:+.3f})",
                     fontsize=13, fontweight="bold", color=tcolor)
        ax.set_xlabel("MJO Phase (Wheeler–Hendon)", fontsize=12, fontweight="bold")
        ax.set_ylabel("TS Genesis Count", fontsize=12, fontweight="bold")
        ax.set_xticks(PHASES)
        ax.grid(alpha=0.3, ls="--")
        ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.axvspan(15, 24, color="green", alpha=0.1, label="Rossby wave arrival (15–24 days)")
    ax.plot(res.lag, res.r2, "b-o", lw=3, ms=8, label="$R^2$ by lag")
    ax.axhline(res.loc[0, "r2"], color="steelblue", ls="--", lw=2, alpha=0.7,
               label=f"Lag 0 ($R^2$={res.loc[0, 'r2']:.3f})")
    ax.axhline(peak.r2, color="darkred", ls="--", lw=2, alpha=0.7,
               label=f"Peak (Lag {int(peak.lag)}, $R^2$={peak.r2:.3f})")
    ax.plot(peak.lag, peak.r2, marker="*", ms=22, color="darkred", mec="black")
    ax.set_xlim(-1, 25)
    ax.set_ylim(0, max(0.8, peak.r2 + 0.1))
    ax.set_xlabel("Lag (days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("$R^2$ (Variance Explained)", fontsize=12, fontweight="bold")
    ax.set_title("MJO–Genesis Relationship vs Lead Time", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(loc="upper left", fontsize=10)

    fig.suptitle(f"Lead-Lag Analysis: TS Genesis Response to MJO Phase ({YEAR_START}–{YEAR_END})\n"
                 f"North Atlantic, active MJO days only (RMM amplitude ≥ 1)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    res.drop(columns="counts").to_csv(out_png.replace(".png", "_stats.csv"), index=False)
    print("Saved", out_png)


if __name__ == "__main__":
    main()
