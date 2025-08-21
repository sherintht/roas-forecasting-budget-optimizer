<!-- ROAS UA Forecasting & Budget Optimizer - README as HTML-styled landing -->
<div style="font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height:1.6; color:#111;">

  <!-- Hero -->
  <div style="background: linear-gradient(135deg,#6EE7F9,#A78BFA 40%,#F472B6); color:#0b1021; border-radius:16px; padding:36px; margin-bottom:28px;">
    <h1 style="margin:0 0 8px 0; font-size:36px;">🎯 ROAS Forecasting & Budget Optimizer for Mobile Game UA</h1>
    <p style="margin:0; font-size:18px;">
      Forecast D7/D30 ROAS, generate scale/cut recommendations, and simulate cross‑channel budget reallocations with an interactive dashboard.
    </p>
    <div style="margin-top:16px; display:flex; gap:10px; flex-wrap:wrap;">
      <img alt="Made with Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
      <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-quantile%20+%20regression-9ACD32?style=for-the-badge"/>
      <img alt="Prophet" src="https://img.shields.io/badge/Prophet-optional-000000?style=for-the-badge"/>
      <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
      <img alt="CI" src="https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?style=for-the-badge&logo=github-actions&logoColor=white"/>
    </div>
  </div>

  <!-- Value props -->
  <div style="display:grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-bottom:28px;">
    <div style="border:1px solid #eee; border-radius:12px; padding:16px; background:#fff;">
      <h3>🚀 End‑to‑end</h3>
      <p>Data → Features → Models → Backtests → Decisions → Dashboard.</p>
    </div>
    <div style="border:1px solid #eee; border-radius:12px; padding:16px; background:#fff;">
      <h3>📈 Forecasting</h3>
      <p>LightGBM point + quantile models; optional Prophet per campaign.</p>
    </div>
    <div style="border:1px solid #eee; border-radius:12px; padding:16px; background:#fff;">
      <h3>🧪 Backtesting</h3>
      <p>Rolling‑origin with MAPE, RMSE, and WAPE to validate decisions.</p>
    </div>
    <div style="border:1px solid #eee; border-radius:12px; padding:16px; background:#fff;">
      <h3>💰 Business Impact</h3>
      <p>Scale/Cut recommendations and budget reallocation simulator with guardrails.</p>
    </div>
  </div>

  <!-- Quick Start -->
  <h2>⚙️ Quick Start</h2>
  <ol>
    <li>
      <strong>Clone & Setup</strong><br/>
      <code>git clone &lt;your_repo_url&gt; && cd roas-ua-forecasting</code><br/>
      <code>python -m venv .venv && source .venv/bin/activate</code><br/>
      <code>pip install --upgrade pip && pip install -r requirements.txt</code>
    </li>
    <li>
      <strong>Run Pipeline</strong><br/>
      <code>python main.py</code><br/>
      Artifacts saved to <code>data/processed</code>.
    </li>
    <li>
      <strong>Launch Dashboard</strong><br/>
      <code>streamlit run app/streamlit_app.py</code>
    </li>
  </ol>

  <!-- Screens / Outputs -->
  <h2>📊 What You’ll See</h2>
  <ul>
    <li>Backtest KPIs: MAPE, RMSE, WAPE.</li>
    <li>Historical vs Predicted D7 ROAS with p05/p95 bands.</li>
    <li>Scale/Cut/Hold recommendations per campaign.</li>
    <li>Budget reallocation: move $ between channels with min spend & max shift caps; see portfolio ROAS uplift.</li>
  </ul>

  <!-- Tech -->
  <h2>🧠 Modeling & Data</h2>
  <div style="border-left:4px solid #a78bfa; padding-left:12px; margin-bottom:18px;">
    <p><strong>Data:</strong> Synthetic MMP‑style daily UA dataset: channel, campaign, country, platform, spend, clicks, installs, revenue_day1/7/30; derived CAC, ROAS_d1/d7/d30.</p>
    <p><strong>Features:</strong> Lags/rolling means, weekday/weekend, holidays, month, CAC, channel/country/platform/campaign_id, bid, target CPI.</p>
    <p><strong>Models:</strong> LightGBM Regression (point) + Quantile (p05/p95) for uncertainty; optional Prophet per campaign for revenue TS → ROAS via projected spend.</p>
    <p><strong>Backtesting:</strong> Rolling‑origin; metrics MAPE, RMSE, WAPE; target D7 ROAS ≤20% MAPE goal (tunable).</p>
    <p><strong>Decisions:</strong> SCALE if forecasted ROAS ≥ threshold, HOLD if p05 below buffer, else CUT.</p>
  </div>

  <!-- Repo structure -->
  <h2>🗂️ Structure</h2>
  <pre style="background:#0b1021; color:#e5e7eb; padding:16px; border-radius:10px; overflow:auto;">
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_generate_data.ipynb
│   └── 02_train_and_backtest.ipynb
├── src/
│   ├── backtest.py
│   ├── business.py
│   ├── data_gen.py
│   ├── features.py
│   ├── tabular.py
│   ├── timeseries.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
  </pre>

  <!-- CI/CD -->
  <h2>🤝 CI/CD (Optional)</h2>
  <p>Automate training and attach artifacts with GitHub Actions. Add <code>.github/workflows/ci.yml</code> to run <code>python main.py</code> on push and upload <code>data/processed/*.csv</code>.</p>

  <!-- Tuning -->
  <h2>🧩 Tuning Tips</h2>
  <ul>
    <li>Improve MAPE: add more lag/roll windows, per‑channel models, tune num_leaves/learning_rate/n_estimators.</li>
    <li>Speed: reduce date range/campaign count in <code>data_gen</code>, lower n_estimators.</li>
    <li>Risk: raise ROAS threshold, increase p05 buffer, adjust max_shift_ratio and min_spend.</li>
  </ul>

  <!-- Requirements -->
  <h2>📦 Requirements</h2>
  <p>Python 3.10–3.11. Key packages: pandas, numpy, scikit‑learn, lightgbm, xgboost, prophet (optional), statsmodels, holidays, streamlit, seaborn, matplotlib.</p>
  <p>If Prophet install fails, try <code>pip install prophet</code> or conda‑forge.</p>

  <!-- Recruiter pitch -->
  <div style="background: #ecfeff; border: 1px solid #a5f3fc; padding:16px; border-radius:12px; margin-top:20px;">
    <h3 style="margin-top:0;">🏁 Recruiter Highlights</h3>
    <ul>
      <li>End‑to‑end ROAS forecasting with realistic synthetic MMP‑style data (no PII).</li>
      <li>Rolling backtests with uncertainty‑aware decisions (quantile models).</li>
      <li>Actionable scale/cut and portfolio budget reallocation with guardrails.</li>
      <li>Interactive Streamlit dashboard for what‑if planning.</li>
    </ul>
  </div>

  <!-- CTA -->
  <div style="margin-top:24px; display:flex; gap:10px; flex-wrap:wrap;">
    <a href="#quick-start" style="background:#111827; color:#fff; padding:10px 14px; border-radius:10px; text-decoration:none;">Run Locally</a>
    <a href="#streamlit" style="background:#16a34a; color:#fff; padding:10px 14px; border-radius:10px; text-decoration:none;">Open Dashboard</a>
    <a href="#cicd" style="background:#2563eb; color:#fff; padding:10px 14px; border-radius:10px; text-decoration:none;">Enable CI</a>
  </div>

  <hr style="margin:28px 0; border:none; border-top:1px solid #eee;"/>

  <!-- Footer -->
  <p style="font-size:13px; color:#6b7280;">
    Built with ❤️ for UA marketers and data scientists. Easily extend to real MMP exports (AppsFlyer/Adjust), connect to BigQuery/Redshift, or add CatBoost/Hyperopt.
  </p>
</div>
