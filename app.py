"""
Cyber Sentinel - dashboard.
Run from the project folder:   streamlit run app.py

Tabs: Network (hybrid IDS) | Email (phishing/spam) | Security events | Metrics
The app only LOADS models saved by the notebooks. It never retrains anything.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from engines import auth, events
from engines import common as C
from engines.network import ModelsNotFound, NetworkEngine
from engines.phishing import PhishingEngine, PhishingModelNotFound

ROOT = Path(__file__).resolve().parent
MODELS, RESULTS = ROOT / "models", ROOT / "results"
DEMO_CSV = ROOT / "data" / "demo_flows.csv"
MAX_ROWS = 300_000
TIERS = [C.TIER_KNOWN, C.TIER_SUSPECT, C.TIER_BENIGN]

st.set_page_config(page_title="Cyber Sentinel", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading network models...")
def load_network() -> NetworkEngine:
    return NetworkEngine(MODELS)


@st.cache_resource(show_spinner="Loading phishing model...")
def load_phishing() -> PhishingEngine:
    return PhishingEngine(MODELS)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x, digits=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}%}"


def log_events(items):
    st.session_state.setdefault("events", [])
    for e in items:
        if e:
            st.session_state["events"].append(e)


# ----------------------------------------------------------------------
# login (password + 6-digit authenticator code)
# ----------------------------------------------------------------------
def login_gate() -> bool:
    if os.environ.get("CS_DISABLE_AUTH") == "1":
        return True
    if st.session_state.get("auth_ok"):
        return True

    st.title("🛡️ Cyber Sentinel")
    st.caption("AI-powered cybersecurity and digital trust platform")
    if not auth.is_configured():
        st.error("2FA is not set up yet. In the project folder run:  `python setup_2fa.py`")
        st.caption("For local development only, set the environment variable CS_DISABLE_AUTH=1 to skip login.")
        return False

    attempts = st.session_state.get("attempts", 0)
    if attempts >= 5:
        st.error("Too many failed attempts. Restart the app to try again.")
        return False

    with st.form("login"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        code = st.text_input("6-digit code from your authenticator app", max_chars=6)
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        ok, msg = auth.verify(user, pwd, code)
        if ok:
            st.session_state["auth_ok"] = True
            st.session_state["user"] = user
            st.rerun()
        else:
            st.session_state["attempts"] = attempts + 1
            st.error(msg)
    return False


# ----------------------------------------------------------------------
# Network tab
# ----------------------------------------------------------------------
def tab_network():
    try:
        eng = load_network()
    except ModelsNotFound as e:
        st.warning(str(e))
        return

    st.subheader("Network traffic scan")
    st.caption("A supervised model catches known attacks. An anomaly model, trained only on normal "
               "traffic, flags unusual flows the supervised model does not recognise.")

    c1, c2 = st.columns([3, 2])
    with c1:
        mode = st.radio("Traffic to scan", ["Demo file (held-out test flows)", "Upload a flow CSV"], horizontal=True)
        upload = st.file_uploader("CICIDS2017-style flow CSV", type="csv") if mode.startswith("Upload") else None
    with c2:
        sim = False
        if eng.has_demo_model:
            sim = st.checkbox(
                f"Simulate a zero-day: use a model that never saw '{eng.demo_family}' attacks",
                help="Scores the same traffic with a supervised model trained without this attack family. "
                     "Watch how much the anomaly layer still catches.")

    if st.button("Scan traffic", type="primary"):
        if mode.startswith("Demo"):
            if not DEMO_CSV.exists():
                st.error("data/demo_flows.csv not found. Run notebook 01 first.")
                return
            raw, src = pd.read_csv(DEMO_CSV), "the demo file"
        else:
            if upload is None:
                st.error("Please choose a CSV file first.")
                return
            raw, src = pd.read_csv(upload, low_memory=False, encoding="latin1"), upload.name
        raw.columns = raw.columns.astype(str).str.strip()
        if len(raw) > MAX_ROWS:
            st.warning(f"File has {len(raw):,} rows; scanning the first {MAX_ROWS:,}.")
            raw = raw.iloc[:MAX_ROWS].copy()
        bar = st.progress(0.0, text="Scanning flows...")
        t0 = time.time()
        try:
            res = eng.predict(raw, use_demo_model=sim, progress=lambda f: bar.progress(f, text="Scanning flows..."))
        except ValueError as e:
            bar.empty()
            st.error(str(e))
            return
        bar.empty()
        st.session_state["net"] = dict(res=res, raw=raw, sim=sim, secs=time.time() - t0, src=src)
        log_events(events.network_events(res, raw, src))

    state = st.session_state.get("net")
    if not state:
        st.info("Choose traffic and press **Scan traffic**.")
        return
    res, raw = state["res"], state["raw"]
    counts = res["tier"].value_counts().reindex(TIERS, fill_value=0)

    if state["sim"]:
        st.warning(f"Zero-day simulation is ON: the supervised model has never seen '{eng.demo_family}'.")
    if res.attrs.get("repaired_rows"):
        st.caption(f"{res.attrs['repaired_rows']:,} rows had missing/infinite values that were set to 0.")

    m = st.columns(5)
    m[0].metric("Flows scanned", f"{len(res):,}")
    m[1].metric("Known attack", f"{counts[C.TIER_KNOWN]:,}")
    m[2].metric("Suspected zero-day", f"{counts[C.TIER_SUSPECT]:,}")
    m[3].metric("Benign", f"{counts[C.TIER_BENIGN]:,}")
    m[4].metric("Scan time", f"{state['secs']:.1f}s")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Flows per tier**")
        st.bar_chart(counts.rename("flows"))
    with right:
        if "Label" in raw.columns:
            st.markdown("**Ground truth vs tier** (available because the file includes labels)")
            ct = pd.crosstab(raw["Label"].astype(str).values, res["tier"].values)
            ct = ct.reindex(columns=TIERS, fill_value=0)
            ct.index.name = "true label"
            st.dataframe(ct, height=300)
            truth_attack = raw["Label"].astype(str).str.upper().ne("BENIGN").to_numpy()
            flagged = (res["tier"] != C.TIER_BENIGN).to_numpy()
            caught = int((truth_attack & flagged).sum())
            st.caption(
                f"Attacks flagged (known or review): {caught:,} of {int(truth_attack.sum()):,} "
                f"({pct(caught / max(truth_attack.sum(), 1), 1)}). "
                f"Benign flows flagged: {int((~truth_attack & flagged).sum()):,} of {int((~truth_attack).sum()):,} "
                f"({pct((~truth_attack & flagged).sum() / max((~truth_attack).sum(), 1))}).")
        else:
            st.caption("No 'Label' column in this file, so no ground-truth comparison is shown.")

    out = pd.DataFrame(index=raw.index)
    if "Destination Port" in raw.columns:
        out["Destination Port"] = raw["Destination Port"]
    if "Label" in raw.columns:
        out["true label"] = raw["Label"]
    out["tier"] = res["tier"]
    out["attack probability"] = res["p_attack"].round(4)
    out["anomaly score"] = res["anomaly_score"].round(4)
    order = out["tier"].map({C.TIER_KNOWN: 0, C.TIER_SUSPECT: 1, C.TIER_BENIGN: 2})
    flagged_view = out.assign(_o=order).sort_values(["_o", "attack probability"], ascending=[True, False]).drop(columns="_o")
    st.markdown("**Most urgent flows**")
    st.dataframe(flagged_view.head(300), height=320)
    st.download_button("Download full results (CSV)", out.to_csv(index=False).encode(),
                       "scan_results.csv", "text/csv")


# ----------------------------------------------------------------------
# Email tab
# ----------------------------------------------------------------------
def _load_example(text: str):
    st.session_state["email_text"] = text


def tab_email():
    try:
        eng = load_phishing()
    except PhishingModelNotFound as e:
        st.warning(str(e))
        return

    st.subheader("Phishing / spam email check")
    st.caption("TF-IDF + logistic regression. The dataset mixes spam and phishing, so the verdict is "
               "'phishing / spam'. The model also shows which words drove its decision.")

    examples = read_json(RESULTS / "phishing_examples.json") or []
    if examples:
        phish = [e["text"] for e in examples if e["label"] == "phishing/spam"]
        legit = [e["text"] for e in examples if e["label"] == "legitimate"]
        b1, b2, _ = st.columns([1, 1, 3])
        if phish:
            b1.button("Load a phishing example", on_click=lambda: _load_example(random.choice(phish)))
        if legit:
            b2.button("Load a legitimate example", on_click=lambda: _load_example(random.choice(legit)))

    text = st.text_area("Paste the email (subject and body)", key="email_text", height=220)
    if st.button("Check email", type="primary"):
        try:
            result = eng.predict(text)
        except ValueError as e:
            st.error(str(e))
            return
        st.session_state["email_result"] = result
        log_events([events.email_event(result, text.strip().replace("\n", " "))])

    result = st.session_state.get("email_result")
    if not result:
        return
    if result["is_phishing"]:
        st.error(f"**{result['verdict']}**  -  probability {result['probability']:.1%}")
    else:
        st.success(f"**{result['verdict']}**  -  phishing probability {result['probability']:.1%}")
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.caption(f"Flag threshold: {result['threshold']:.3f} (tuned on validation data for a 1% false-alarm rate).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Words pushing towards phishing / spam**")
        w = result["words_toward_phishing"]
        st.dataframe(pd.DataFrame(w, columns=["word", "weight"]).round(3) if w else pd.DataFrame({"word": []}),
                     hide_index=True)
    with c2:
        st.markdown("**Words pushing towards legitimate**")
        w = result["words_toward_legitimate"]
        st.dataframe(pd.DataFrame(w, columns=["word", "weight"]).round(3) if w else pd.DataFrame({"word": []}),
                     hide_index=True)


# ----------------------------------------------------------------------
# Events tab
# ----------------------------------------------------------------------
def tab_events():
    st.subheader("Security events")
    st.caption("One timeline for both engines. Severity comes from simple, documented rules "
               "(it is not a learned risk model).")
    evs = st.session_state.get("events", [])
    if not evs:
        st.info("No events yet. Scan some traffic or check an email.")
    else:
        df = pd.DataFrame(evs).iloc[::-1].reset_index(drop=True)
        sev = st.multiselect("Show severities", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        df = df[df["severity"].isin(sev)]
        icons = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟡 Low"}
        view = df.assign(severity=df["severity"].map(icons))
        st.dataframe(view[["time", "severity", "source", "title", "detail"]],
                     hide_index=True)
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("Download CSV", df.to_csv(index=False).encode(), "security_events.csv", "text/csv")
        if c2.button("Clear events"):
            st.session_state["events"] = []
            st.rerun()
    with st.expander("How severity is decided"):
        st.table(pd.DataFrame(events.RULES, columns=["Engine", "Condition", "Severity"]))


# ----------------------------------------------------------------------
# Metrics tab
# ----------------------------------------------------------------------
def tab_metrics():
    st.subheader("Evidence: measured on held-out test data")
    net = read_json(RESULTS / "results.json")
    ph = read_json(RESULTS / "phishing_results.json")
    if not net and not ph:
        st.info("No results yet. Run notebooks 01 and 02 first.")
        return

    if net:
        st.markdown("### Network IDS")
        f = net["fusion"]
        table = pd.DataFrame([
            ["Random Forest, default threshold (0.5)", net["rf_default"]["FPR"], net["rf_default"]["recall"], net["rf_default"]["precision"]],
            ["Random Forest, tuned threshold", net["rf_tuned"]["FPR"], net["rf_tuned"]["recall"], net["rf_tuned"]["precision"]],
            ["Anomaly layer alone", net["anomaly_alone"]["FPR"], net["anomaly_alone"]["recall"], net["anomaly_alone"]["precision"]],
            ["Fused: known-attack tier", f["FPR_known_tier"], f["recall_known_tier"], f["precision_known"]],
            ["Fused: any alert (known + review)", f["FPR_any_alert"], f["recall_any_alert"], np.nan],
        ], columns=["System", "False positive rate", "Recall", "Precision"])
        for col in table.columns[1:]:
            table[col] = table[col].map(pct)
        st.dataframe(table, hide_index=True)

        d = net["dataset"]
        st.caption(f"{d['rows_after_cleaning']:,} flows after removing duplicates and bad rows; "
                   f"{d['features']} features; test split = {d['test']:,} flows.")

        c1, c2 = st.columns(2)
        if (RESULTS / "pr_curve.png").exists():
            c1.image(str(RESULTS / "pr_curve.png"), caption="Precision-recall and detection vs false-alarm budget")
        if (RESULTS / "zero_day.png").exists():
            c2.image(str(RESULTS / "zero_day.png"), caption="Zero-day: recall on attack families never seen in training")
        zd = pd.DataFrame(net["zero_day"])
        for col in ["supervised_recall", "anomaly_recall", "fused_recall", "supervised_FPR", "fused_FPR"]:
            zd[col] = zd[col].map(pct)
        st.markdown("**Zero-day experiment (leave-one-attack-family-out)**")
        st.dataframe(zd, hide_index=True)

    if ph:
        st.markdown("### Phishing / spam engine")
        t, dflt = ph["tuned"], ph["default"]
        pt = pd.DataFrame([
            ["Default threshold (0.5)", dflt["accuracy"], dflt["precision"], dflt["recall"], dflt["FPR"]],
            ["Tuned threshold", t["accuracy"], t["precision"], t["recall"], t["FPR"]],
        ], columns=["Setting", "Accuracy", "Precision", "Recall", "False positive rate"])
        for col in pt.columns[1:]:
            pt[col] = pt[col].map(pct)
        st.dataframe(pt, hide_index=True)
        loso = pd.DataFrame(ph["leave_one_source_out"])
        for col in ["share_phishing", "accuracy", "recall", "FPR", "ROC_AUC"]:
            loso[col] = loso[col].map(lambda v: pct(v) if col != "ROC_AUC" else ("n/a" if pd.isna(v) else f"{v:.3f}"))
        st.markdown("**Leave-one-source-out** (train without a source, test on it)")
        st.dataframe(loso, hide_index=True)
        if (RESULTS / "phishing_loso.png").exists():
            st.image(str(RESULTS / "phishing_loso.png"))

    with st.expander("Limitations (read before presenting)"):
        st.markdown(
            "- Network models were trained on **CICIDS2017**, a 2017 lab network. Performance on other "
            "networks is usually lower (domain shift).\n"
            "- The demo scores flow files; it does not capture live packets.\n"
            "- Thresholds were tuned on this dataset and need recalibrating elsewhere.\n"
            "- The email dataset mixes spam and phishing and comes from sources with different styles; "
            "the leave-one-source-out table shows how much of the accuracy depends on that.\n"
            "- Event severity is rule-based, not a learned risk model.")


# ----------------------------------------------------------------------
def main():
    if not login_gate():
        st.stop()

    st.title("🛡️ Cyber Sentinel")
    st.caption("AI-powered cybersecurity and digital trust platform")

    with st.sidebar:
        if st.session_state.get("user"):
            st.write(f"Signed in as **{st.session_state['user']}**")
            if st.button("Sign out"):
                for k in ("auth_ok", "user", "attempts"):
                    st.session_state.pop(k, None)
                st.rerun()
        st.markdown("**Engines**")
        st.markdown("- Network: Random Forest + Isolation Forest\n- Email: TF-IDF + Logistic Regression")
        meta = read_json(MODELS / "ids_meta.json")
        if meta:
            st.caption(f"Network thresholds: T1={meta['t1']:.3f}, T2={meta['t2']:.3f}")

    t1, t2, t3, t4 = st.tabs(["Network", "Email", "Security events", "Metrics"])
    with t1:
        tab_network()
    with t2:
        tab_email()
    with t3:
        tab_events()
    with t4:
        tab_metrics()


main()
