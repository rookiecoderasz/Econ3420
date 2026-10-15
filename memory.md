# ECON 3420A Assignment 1 - Project Memory

## Rules
- Do not read data files (token budget). Use only code/templates.
- Deliverables: `portfolio_model.py`, `studentID_assignment_one.qmd`, rendered PDF (same name).
- Python: `C:/Users/Andy/AppData/Local/Python/pythoncore-3.14-64/python.exe` (3.14.7). All `requirements.txt` packages import OK.
- Quarto: NOT installed yet (needed to render PDF via typst).
- Assignment PDF text extracted with `pypdf` (installed).

## Functions to implement in portfolio_model.py
- Q2 `estimate_index_model`: OLS X_i = a + b X_M + e; return alpha, beta, R2, residuals.
- Q3 `single_index_covariance`: var_i = b_i^2 var_M + s2(e_i), s2(e)=sum(e^2)/(T-2); cov_ij = b_i b_j var_M.
- Q4 `optimize_risky_portfolio`: max Sharpe, w>=0, sum=1; annualize x12; compare with 20% equal weight.
- Q5 `infer_risk_aversion`: delta=mu_P-Rf, G=delta/var_P, gamma_m=m*G, c_m=mu_P-gamma_m*var_P/2, W_m=10000(1+c_m); delta<=0 exception.
- Q6 `allocate_portfolio`: w*=delta/(gamma*var_P); F return=Rf+w*delta, vol=w*sigma_P; delta<=0 -> 100% safe.

## Plan / status
- [x] Step 0: setup (packages verified; memory.md created)
- [ ] Step 0: install Quarto; rename template to `<studentID>_assignment_one.qmd` (student ID needed)
- [ ] Step 1: beta prediction (a), data sources (c) in QMD
- [ ] Step 2: Q1 stocks/returns table
- [ ] Step 3: implement 5 functions
- [ ] Step 4: Q2-Q4 in QMD, render (saves portfolio P)
- [ ] Step 5: Q5 run `risk_aversion_quiz.py` in terminal (interactive)
- [ ] Step 6: Q6 prediction BEFORE half-premium test; AI evidence table (b)
- [ ] Step 7: render PDF, remove all `[WRITE]`/`[INSERT]` placeholders

## Template notes
- Remove `#| eval: false` per cell after its function is done.
- Config: stocks 0700.HK, 0005.HK, 0939.HK, 0388.HK, 0016.HK; benchmark 2800.HK; HKD; window 2021-10 to 2026-09 (practice data).
