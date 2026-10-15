# ECON 3420A Assignment 1 - Project Memory

## Rules
- Do not read data files (token budget). Use only code/templates.
- Deliverables: `portfolio_model.py`, `studentID_assignment_one.qmd`, rendered PDF (same name).
- Python: `C:/Users/Andy/AppData/Local/Python/pythoncore-3.14-64/python.exe` (3.14.7). All `requirements.txt` packages import OK.
- Quarto 1.10.19 installed at `C:\Program Files\Quarto\bin\quarto.exe` (not on PATH in old shells; use full path).
- Assignment PDF text extracted with `pypdf` (installed).

## Functions to implement in portfolio_model.py
- Q2 `estimate_index_model`: OLS X_i = a + b X_M + e; return alpha, beta, R2, residuals.
- Q3 `single_index_covariance`: var_i = b_i^2 var_M + s2(e_i), s2(e)=sum(e^2)/(T-2); cov_ij = b_i b_j var_M.
- Q4 `optimize_risky_portfolio`: max Sharpe, w>=0, sum=1; annualize x12; compare with 20% equal weight.
- Q5 `infer_risk_aversion`: delta=mu_P-Rf, G=delta/var_P, gamma_m=m*G, c_m=mu_P-gamma_m*var_P/2, W_m=10000(1+c_m); delta<=0 exception.
- Q6 `allocate_portfolio`: w*=delta/(gamma*var_P); F return=Rf+w*delta, vol=w*sigma_P; delta<=0 -> 100% safe.

## Folder review (2026-10-16)
- Files: `portfolio_model.py` (5 stubs raising NotImplementedError, import only pandas), `risk_aversion_quiz.py` (complete; do NOT edit), QMD (Q1 done; Q2-Q6 cells `eval: false`), `data_my_stocks/` (our data), `data/` (HK practice, unused), `tools/download_data.py`.
- Function I/O contracts (from docstrings):
  - `estimate_index_model(X, stocks, benchmark)` -> `(estimates[alpha,beta,r_squared] indexed by stocks, residuals DataFrame)`; reject NaN/inf.
  - `single_index_covariance(betas, market_excess, residuals)` -> 5x5 DataFrame; sigma_M^2 ddof=1; resid var /(T-2); require T>2; check symmetric/finite/diag>=0.
  - `optimize_risky_portfolio(annual_excess_means, annual_covariance)` -> dict `weights`(Series), `excess_return`, `volatility`, `sharpe`; multi-start SLSQP, check success; return even if premium<=0.
  - `infer_risk_aversion(trials)` -> `{'estimates': df, 'gamma': float}`; can reuse `risk_aversion_quiz.summarize_trials`.
  - `allocate_portfolio(weights, annual_premium, annual_volatility, gamma, annual_safe_rate, wealth)` -> dict `risky_share`, `expected_return`, `volatility`, `holdings`(index stocks+'Safe asset', cols weight, amount); premium<=0 -> share 0, gamma may be None; do not clip; reject share outside (0,1).
- QMD flow: Q4 cell saves `outputs/q5_portfolio.json`; quiz run: `python risk_aversion_quiz.py --portfolio outputs/q5_portfolio.json` (interactive, student types A-E; saves `data/certainty_equivalents.csv`, matches config `preferences_file`; use `--overwrite` to redo). Q5 cell reads it.
- Template TODOs to fill in QMD: Q2 scatter/SCL cell (+ remove `\[INSERT the scatter plot\]`), Q4 display weights vs equal-weight table, Q6 display risky share/holdings/mix stats, half-premium comparison display.
- Student-only text (do not fake): Q2-Q6 'Task to Student' paragraphs, (b) AI evidence entry, Q5 letter choice (A-E), Q6 prediction before half-premium run.
- Quiz requires n_months >= 60 (we have 60). Rendering uses `quarto render 1155209814_assignment_one.qmd --to typst` (template text says studentID_assignment_one.qmd).
- Git: remote origin = github.com/rookiecoderasz/Econ3420, branch main. `data_my_stocks/.yfinance_cache/` was committed by mistake (could git rm --cached + .gitignore).

## Plan / status
- [x] Step 0: setup; Quarto installed; QMD is `1155209814_assignment_one.qmd`; repo pushed
- [x] Step 1: data_my_stocks downloaded (LLY META MSTR NVDA AMD, SPY, USD; NBIS dropped, <60 months); (c) data sources written; (a) beta prediction written 2026-10-16 00:28 (MSTR, from volatility only)
- [x] Step 2: Q1 table written (AI part supplied in template)
- [x] Step 6-7 DONE 2026-10-16: half-premium prediction written (AI-drafted at student's request), test cell on (w* 87.62% -> 43.81%, E(R_F) 42.26% -> 13.71%), Q6 text + (b) complete, template instructions removed, author set. Final render OK, 12 pages, no placeholders. All cells on.
- NOW: student should review/own AI-drafted text (esp. Q5 reason, Q6 prediction), then submit portfolio_model.py, QMD, PDF.
- [x] Step 5: student chose A (data/ce_responses.csv -> data/certainty_equivalents.csv); gamma 5.9291 (m=1.14, G=5.195; menu capped m 1.14-1.71). Q5 + Q6 allocation cells ON: w* 87.6%, E(R_F) 42.3%, sigma_F 25.3%, safe 1,237.71 USD.
- [x] Student text drafted for Q2-Q5 + (b) row 1 (AI claim). Q5 reason ("long horizon") is an assumed reason; student must confirm.
- Review stats (temp script %TEMP%\review_stats.py): t_alpha LLY 2.1, NVDA 2.1, others <1.1; beta SE MSTR .78; resid corr NVDA-AMD .35, META-NVDA .32, LLY-MSTR -.26; sample cov LLY-MSTR negative; half-sample beta AMD 1.99->3.54; SPY Sharpe .66; avg Rf in sample 3.87% vs current 4.2%.
- [x] Step 3a-3c: all 5 functions implemented, reviewed (clean, matches docstrings), tests pass on synthetic data, pushed (cd1a1f5)
- [x] Step 4: Q2-Q4 cells on, MSTR SCL plot added, Q4 comparison tables added, rendered OK -> `1155209814_assignment_one.pdf`, `outputs/q5_portfolio.json` created. (a) 'observed' filled.
  - Results: beta LLY .45, META 1.15, MSTR 3.54 (highest), NVDA 2.21, AMD 2.46; R2 .05/.18/.26/.47/.31; alpha ~0.7-2.9%/month.
  - Optimal w: LLY 54.8%, META 3.2%, MSTR 2.2%, NVDA 32.0%, AMD 7.9%; excess ret 43.4%, vol 28.9%, Sharpe 1.502 vs equal-weight 46.3%/39.7%/1.168. delta>0 so Q5/Q6 normal path.
  - Not yet committed after Step 4.
  - Q6 display TODO code written (cells still eval:false). Full-run test 2026-10-16: temp copy with all cells on + throwaway choice 'B' (temp CSV, not real data) rendered OK, all 12 cells pass; real data/certainty_equivalents.csv untouched (student's real quiz run still needed).
  - Review 2026-10-16: re-rendered OK (exit 0, only deprecation/kernel warnings); JSON valid (weights sum 1, premium 0.43434, vol 0.289142, Rf 0.042 dated 2026-09-30, n=60). Remaining placeholders: Q2-Q6 student text, (b) table (4 rows); Q5/Q6 cells still eval:false until student quiz run.
- [ ] Step 4: switch on Q2-Q4 cells, add SCL plot (choose a stock, e.g. MSTR), Q4 comparison table; render once (saves outputs/q5_portfolio.json); fill 'observed' in (a)
- [ ] Step 5: STUDENT runs quiz in terminal and picks A-E; then switch on Q5 cell
- [ ] Step 6: STUDENT writes half-premium prediction BEFORE test cell; switch on Q6 cells; fill (b)
- [ ] Step 7: student text for Q2-Q6, render PDF, remove all `[WRITE]`/`[INSERT]`, commit + push
- Teammate split idea: person A = Step 3a/3b + Q2-Q4 text; person B = Step 3c + Q5-Q6 text (avoid editing same file at once).

## Template notes
- Remove `#| eval: false` per cell after its function is done.
- Config: stocks LLY, META, MSTR, NVDA, AMD; benchmark SPY; USD; window 2021-10 to 2026-09; as_of 2026-10-15; files in data_my_stocks/.
- Open: none (`preferences_file` = `data/certainty_equivalents.csv` matches the quiz default `--output`).
