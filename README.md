# hank1.py — Simulating Household Responses to Trade-Policy Uncertainty

This script implements a partial-equilibrium Heterogeneous Agent New Keynesian (HANK) model to study the impact of trade-policy uncertainty (TPU) on U.S. household behavior. The model is designed to replicate key features of the U.S. income and asset distribution, and to simulate impulse responses following TPU shocks calibrated to the Baker–Bloom–Davis index.

## Key Features
- Income Process: Idiosyncratic permanent and transitory labor income risk, with unemployment shocks.
- Shock Structure: Regime-switching increase in transitory income volatility and unemployment risk representing TPU events.
- Calibration: Matches moments from the Survey of Consumer Finances (SCF), including income variance and liquid asset holdings.
- Simulation Horizon: 400 quarters (steady state) + 20 quarters (post-shock impulse response).
- Outputs:
  - Steady-state aggregates (consumption, liquid assets)
  - Time path of aggregate consumption and assets after +1σ and +2σ TPU shocks
  - Impulse response figures (PDF and PNG)
  - Output CSV for further analysis

## Dependencies
- Python ≥ 3.11
- [HARK](https://github.com/econ-ark/HARK) ≥ v0.16
- NumPy, Matplotlib, Pandas

## Usage
```bash
python hank1.py
