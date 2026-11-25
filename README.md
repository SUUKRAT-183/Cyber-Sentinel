# Cyber Sentinel

AI-powered cybersecurity and digital trust prototype with two engines behind one secure dashboard.

| Engine | What it does | Models | Data |
|---|---|---|---|
| **Network IDS** (deep one) | Scores network flows into **Known attack / Suspected zero-day / Benign** | Random Forest (known attacks) + Isolation Forest trained on benign traffic only (unusual behaviour), fused with tuned thresholds | CICIDS2017 |
| **Email** | Phishing / spam verdict with the words that drove it | TF-IDF + Logistic Regression | Enron, Ling, CEAS-08, SpamAssassin, Nazario, Nigerian Fraud |
| **Login** | Password + 6-digit authenticator code (TOTP) | a security feature, not ML | none |
| **Security events** | One timeline for both engines with rule-based severity (High / Medium / Low) | rules, not a learned model | none |

## Folder layout

```
cyber-sentinel/
├── app.py                      # Streamlit dashboard (login + 4 tabs)
├── setup_2fa.py                # run once: creates login + QR code
├── make_sample.py              # optional: small CICIDS sample for a quick dry run
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml
├── engines/
│   ├── __init__.py
│   ├── common.py               # shared cleaning + model definitions (used by notebooks AND app)
│   ├── network.py              # NetworkEngine: loads models, scores flows into tiers
│   ├── phishing.py             # PhishingEngine: verdict + explanation
│   ├── auth.py                 # password + TOTP
│   └── events.py               # rule-based severity, combined feed
├── notebooks/
│   ├── 01_network_ids.ipynb    # train + evaluate network engine, zero-day experiment
│   └── 02_phishing_nlp.ipynb   # train + evaluate phishing engine, leave-one-source-out
├── data/
│   ├── cicids/                 # YOU put the 8 CICIDS2017 CSV files here
│   ├── phishing/               # YOU put the 7 phishing CSV files here
│   └── demo_flows.csv          # created by notebook 01
├── models/                     # created by the notebooks
└── results/                    # created by the notebooks (tables, charts, json)
```

## Setup (once)

1. Install Python 3.10 - 3.12.
2. In the `cyber-sentinel` folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)        source .venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   ```
3. Put the data in place:
   - the 8 CICIDS2017 `.csv` files into `data/cicids/`
   - the phishing archive contents (`Enron.csv`, `Ling.csv`, `CEAS_08.csv`, `SpamAssasin.csv`, `Nazario.csv`, `Nigerian_Fraud.csv`; `phishing_email.csv` is not needed) into `data/phishing/`

## Run order

| Step | Command / action | Time (typical laptop) | Creates |
|---|---|---|---|
| 1 | `jupyter notebook`, open `notebooks/01_network_ids.ipynb`, **Kernel > Restart & Run All** | 20-40 min | `models/ids_*`, `results/*`, `data/demo_flows.csv` |
| 2 | Open `notebooks/02_phishing_nlp.ipynb`, **Restart & Run All** | 5-15 min | `models/phishing_*`, `results/phishing_*` |
| 3 | `python setup_2fa.py` and scan the QR code with an authenticator app | 2 min | `auth_config.json`, `2fa_qr.png` |
| 4 | `streamlit run app.py` | instant | the dashboard at http://localhost:8501 |

Memory: the full CICIDS2017 needs roughly 6-8 GB of free RAM during notebook 01. If it runs out of memory, lower `TRAIN_CAP_PER_LABEL` and `ANOMALY_TRAIN_ROWS` in the settings cell, or do the dry run below.

### Optional 2-minute dry run for notebook 01
```
python make_sample.py
set CICIDS_DIR=data\cicids_sample        (Windows cmd)     export CICIDS_DIR=data/cicids_sample   (Mac/Linux)
jupyter notebook                         (start it from the same window so it sees the variable)
```
Numbers from the sample are not final; delete `models/` and `results/` content and re-run on the full data before presenting.

### Development shortcut
`set CS_DISABLE_AUTH=1` (or `export ...`) skips the login. Never use it for the real demo.

## What each notebook measures

**01 Network IDS**
- Cleaning: removes about 9% exact duplicate rows, bad rows (inf/NaN/negative duration), a duplicated column and garbled labels.
- Split first (60/20/20 stratified), so scaling, feature selection and models never see validation/test data.
- False positive rate (FPR) for every setting, precision-recall curve, per-attack recall.
- Isolation Forest (benign only) and the three-tier fusion; thresholds are chosen on validation data for a 1% false-alarm budget per layer.
- **Zero-day experiment:** for each attack family (DoS, DDoS, PortScan, Patator, Bot, WebAttack) it is removed from supervised training and then tested.
- The app's "Simulate a zero-day" box uses the experiment model for the family where the anomaly layer helps most.

**02 Phishing**
- Cleans text, removes duplicates, splits before fitting TF-IDF, tunes the threshold for a 1% false-alarm rate on legitimate email.
- **Leave-one-source-out:** shows how much accuracy depends on the style of each email source.

## 15-minute demo script (suggested)

| Min | Do | Say |
|---|---|---|
| 0-3 | Slides: problem, architecture | Why false alarms and unseen attacks matter |
| 3-4 | Log in with password + authenticator code | Security feature of the platform |
| 4-7 | **Network tab**, demo file, Scan. Show tier counts and the "ground truth vs tier" table | Known attacks, review queue, false alarms |
| 7-8 | Tick **Simulate a zero-day**, Scan again | The supervised model has never seen this attack; the anomaly layer still raises some of it for review |
| 8-10 | **Email tab**: load a phishing example, then a legitimate one | Verdict plus the words that drove it |
| 10-11 | **Security events** tab | One timeline, severity from documented rules |
| 11-14 | **Metrics** tab: FPR table, zero-day table, leave-one-source-out | Honest results, including where it fails |
| 14-15 | Limitations and future scope | See below |

Practise this twice on the exact machine you will present from. Keep a screen recording as a backup.

## Limitations (say these before a judge asks)
- Trained on CICIDS2017, a 2017 lab network. Other networks will behave differently; the anomaly layer is the part that can be recalibrated on a network's own benign traffic.
- The demo scans flow CSV files. Live packets would first need a flow extractor such as CICFlowMeter.
- The anomaly layer helps for some attack families (the DoS family) and not for others (e.g. port scans); the zero-day table shows exactly which.
- The email dataset mixes spam and phishing, and its sources differ in style.
- Event severity is rule-based. The network and email datasets share no users or events, so a validated combined risk model is not possible.

## Future scope
Live capture with CICFlowMeter; retraining the anomaly layer on a target network's own benign traffic; sending alerts to a SIEM or chat tool; an analyst feedback loop to cut false positives over time; drift monitoring.

## Troubleshooting
- **"Network models not found"** in the app: run notebook 01 first (and 02 for the Email tab).
- **"missing required column(s)"**: the uploaded CSV must be CICIDS2017 / CICFlowMeter style.
- **ModuleNotFoundError: engines**: start Jupyter and Streamlit from the project folder (or the `notebooks` folder for Jupyter).
- **Model fails to load after upgrading scikit-learn**: re-run the notebooks in the same environment you run the app in.
- **MemoryError in notebook 01**: lower the caps in the settings cell, close other programs, or use the dry-run sample.
