#!/usr/bin/env python3
# Fixed version: Persistent TPU shock HARK IRF simulation

from __future__ import annotations
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import random
from pandas_datareader.data import DataReader
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

# ---------------------------------------------------------------------
# 0. Helpers
# ---------------------------------------------------------------------
def broadcast_path(x, T: int):
    if np.isscalar(x):
        return [[x]] * T
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 1:
            return [[x[0]]] * T
        assert len(x) == T, "Path length mismatch"
        return [[float(z)] for z in x]
    raise TypeError("Unsupported type for broadcast_path")

# ---------------------------------------------------------------------
# 1. EPU shock scale from FRED
# ---------------------------------------------------------------------
SERIES = "EPUTRADE"
START, END = "1985-01-01", "2025-04-30"
print("Pulling EPUTRADE series from FRED ...")
log_epu = np.log(DataReader(SERIES, "fred", START, END)[SERIES].dropna())
sigma_epu = log_epu.std(ddof=0)
print(f"σ(log EPUTRADE) = {sigma_epu:.4f}")

# ---------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------
CRRA, Beta = 2.0, 0.988
Nagents = 30000
PermShkStd = 0.06
TranShkStd = 0.20
u0, kappa = 0.04, 0.02
LivPrb = [0.993]
Rfree = [1.0125]
T_sim_long = 400
IRF_HORIZON = 20
SEED = 47

consumer_dict = dict(
    CRRA=CRRA,
    DiscFac=Beta,
    Nagents=Nagents,
    AgentCount=Nagents,
    PermShkStd=[PermShkStd],
    TranShkStd=[TranShkStd],
    UnempPrb=u0,
    IncUnemp=0.3,
    BoroCnstArt=-4.0,
    LivPrb=[0.993],
    Rfree=[1.0125],
    PermGroFac=[1.001],
    T_total=120,
    T_retire=120,
    T_sim=T_sim_long,
)

print('DEBUG consumer_dict:')
for k, v in consumer_dict.items():
    print(f'  {k}: {v} (type: {type(v)})')

print("Solving low-tariff steady state ...")
agent = IndShockConsumerType(**consumer_dict)
agent.solve()
agent.track_vars = ['cNrm', 'aNrm']
random.seed(SEED)
np.random.seed(SEED)
agent.initialize_sim()
agent.simulate()

C0 = np.mean(agent.history['cNrm'][-1])
A0 = np.mean(agent.history['aNrm'][-1])
assert A0 > 1e-6, "Mean assets ≈0 -- check calibration."
print(f"Steady-state aggregates: C0 = {C0:.4f}   A0 = {A0:.4f}")

# ---------------------------------------------------------------------
# 3. Run IRF
# ---------------------------------------------------------------------
def run_tpu_irf(k_sigma=1.0,
                horizon: int = IRF_HORIZON,
                persistence: float = 0.05):
    ag = IndShockConsumerType(**consumer_dict)
    ag.solve()
    ag.track_vars = ['cNrm', 'aNrm']

    np.random.seed(SEED + int(1000 * k_sigma))
    regime = [1]
    for _ in range(horizon - 1):
        prev = regime[-1]
        if prev == 1:
            regime.append(0 if np.random.rand() < persistence else 1)
        else:
            regime.append(0)
    print(f"[DEBUG] Regime path (k_sigma={k_sigma}): {regime}")

    tran_bump = k_sigma * sigma_epu / 10
    tran_path = [TranShkStd + tran_bump if r == 1 else TranShkStd for r in regime]
    unemp_path = [u0 + kappa * k_sigma if r == 1 else u0 for r in regime]
    inc_unemp_path = [0.3 for _ in regime]

    ag.TranShkStd = tran_path
    ag.UnempPrb = unemp_path
    ag.IncUnemp = inc_unemp_path
    ag.time_vary = ['UnempPrb', 'TranShkStd', 'IncUnemp']
    ag.update_income_process()

    ag.T_sim = horizon
    ag.initialize_sim()
    ag.simulate()

    c = np.array([np.mean(ag.history['cNrm'][t]) for t in range(horizon)])
    a = np.array([np.mean(ag.history['aNrm'][t]) for t in range(horizon)])
    print(f"[DEBUG] Markov: σ_e(0) = {tran_path[0]:.4f}, u(0) = {unemp_path[0]:.4f}, mean aNrm(0) = {a[0]:.4f}")
    return c, a

# ---------------------------------------------------------------------
# 4. Run and export
# ---------------------------------------------------------------------
print("Running impulse responses ...")
c1, a1 = run_tpu_irf(1.0)
c2, a2 = run_tpu_irf(2.0)

C_pct1 = 100 * (c1 / C0 - 1)
C_pct2 = 100 * (c2 / C0 - 1)
A_pct1 = 100 * (a1 / A0 - 1)
A_pct2 = 100 * (a2 / A0 - 1)

Path("figures").mkdir(exist_ok=True)

def make_plot(series, title, fname, ylabel):
    plt.figure(figsize=(7, 5))
    for arr, lab, mark in series:
        plt.plot(arr, label=lab, marker=mark, markersize=6, linewidth=1.8, alpha=0.85)
    plt.axhline(0, ls='--', lw=0.6, c='gray')
    plt.title(title)
    plt.xlabel("Quarters after shock")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=300)

make_plot([(C_pct1, "+1 σ", 'o'), (C_pct2, "+2 σ", 's')],
          "Consumption IRF (% dev.)", "irf_C_pct.png",
          "% deviation from steady state")

make_plot([(A_pct1, "+1 σ", 'o'), (A_pct2, "+2 σ", 's')],
          "Liquid-asset IRF (% dev.)", "irf_A_pct.png",
          "% deviation from steady state")

make_plot([(a1, "a(t) +1 σ", 'o'), (a2, "a(t) +2 σ", 's')],
          "Raw asset paths", "irf_A_raw.png",
          r"Mean $a^{\mathrm{nrm}}$")

pd.DataFrame({
    "q": np.arange(IRF_HORIZON),
    "C_pct_1σ": C_pct1,  "C_pct_2σ": C_pct2,
    "A_pct_1σ": A_pct1,  "A_pct_2σ": A_pct2,
    "a_raw_1σ": a1,      "a_raw_2σ": a2,
}).to_csv("figures/irf_tpu_markov_hark016.csv", index=False)

print("IRF figures and CSV saved in ./figures/")
