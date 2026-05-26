"""
app.py

Behavioral Session Monitor — Streamlit application.
Simulates a fraud analyst console for continuous behavioral authentication.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.profiler import compute_eer, build_profile
from src.scoring import optimal_threshold

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Behavioral Session Monitor",
    page_icon="🛡️",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
KEYSTROKE_FEATURES = [
    'dwell_median', 'dwell_mean', 'dwell_std', 'dwell_max',
    'flight_median', 'flight_min', 'click_dwell_mean'
]

FEATURE_LABELS = {
    'dwell_median':     'Dwell Median',
    'dwell_mean':       'Dwell Mean',
    'dwell_std':        'Dwell Std Dev',
    'dwell_max':        'Dwell Max',
    'flight_median':    'Flight Median',
    'flight_min':       'Flight Min',
    'click_dwell_mean': 'Click Dwell Mean'
}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/features_kmt.csv')
    return df

# ── Train model for selected user ─────────────────────────────────────────────
@st.cache_resource
def train_model(user_id):
    df = load_data()
    user_data = df[df['user_id'] == user_id]
    legitimate = user_data[user_data['label'] == 'legitimate'][KEYSTROKE_FEATURES].values
    impostor   = user_data[user_data['label'] == 'impostor'][KEYSTROKE_FEATURES].values

    model = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    model.fit(legitimate)

    all_features = np.vstack([legitimate, impostor])
    all_labels   = np.array([1]*len(legitimate) + [0]*len(impostor))
    scores       = -model.decision_function(all_features)
    _, frr, far, thresholds = compute_eer(all_labels, scores)

    low_frr_thresh = optimal_threshold(frr, far, thresholds, strategy='low_frr')
    low_far_thresh = optimal_threshold(frr, far, thresholds, strategy='low_far')
    centroid       = legitimate.mean(axis=0)

    return model, low_frr_thresh, low_far_thresh, centroid


def get_risk_tier(score, low_frr_thresh, low_far_thresh):
    if score <= low_frr_thresh:
        return 'CLEAR', '#2ecc71', '✅'
    elif score >= low_far_thresh:
        return 'ESCALATE', '#e74c3c', '🚨'
    else:
        return 'REVIEW', '#f39c12', '⚠️'


def score_bar(score, low_frr, low_far):
    fig, ax = plt.subplots(figsize=(6, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Background zones
    ax.barh(0.5, low_frr,            left=0,       height=0.6, color='#2ecc71', alpha=0.3)
    ax.barh(0.5, low_far - low_frr,  left=low_frr, height=0.6, color='#f39c12', alpha=0.3)
    ax.barh(0.5, 1 - low_far,        left=low_far, height=0.6, color='#e74c3c', alpha=0.3)

    # Zone labels inside bar
    ax.text(low_frr / 2,             0.5, 'CLEAR',    ha='center', va='center',
            fontsize=7, color='#27ae60', fontweight='bold')
    ax.text((low_frr + low_far) / 2, 0.5, 'REVIEW',   ha='center', va='center',
            fontsize=7, color='#e67e22', fontweight='bold')
    ax.text((low_far + 1) / 2,       0.5, 'ESCALATE', ha='center', va='center',
            fontsize=7, color='#c0392b', fontweight='bold')

    # Score marker
    norm_score = np.clip(score, 0, 1)
    ax.axvline(norm_score, color='black', linewidth=3)

    ax.set_yticks([])
    ax.set_xticks([0, low_frr, low_far, 1])
    ax.set_xticklabels(['0', '', '', '1'], fontsize=7)
    ax.tick_params(axis='x', pad=4)
    ax.set_title('Anomaly Score', fontsize=9)
    fig.tight_layout()
    return fig


def deviation_chart(session_features, centroid):
    deviations = session_features - centroid
    labels     = [FEATURE_LABELS[f] for f in KEYSTROKE_FEATURES]
    colors     = ['#e74c3c' if d > 0 else '#2ecc71' for d in deviations]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, deviations, color=colors, alpha=0.8, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Feature Deviation from Legitimate Centroid', fontsize=9)
    ax.set_xlabel('Deviation (raw units)', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    return fig


# ── Session state init ────────────────────────────────────────────────────────
if 'session_idx' not in st.session_state:
    st.session_state.session_idx = 0
if 'review_queue' not in st.session_state:
    st.session_state.review_queue = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'processed': 0, 'clear': 0, 'review': 0,
        'escalate': 0, 'confirmed_fraud': 0, 'false_alarm': 0
    }
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'last_user' not in st.session_state:
    st.session_state.last_user = None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Behavioral Session Monitor")
st.caption("Continuous authentication console — flags anomalous sessions for human review")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
df = load_data()
user_ids = sorted(df['user_id'].unique())

with st.sidebar:
    st.header("Configuration")
    selected_user = st.selectbox("Monitored User", user_ids, index=0)
    st.caption("Sessions are fed in mixed order — legitimate and impostor interleaved.")
    st.divider()
    st.markdown("**Risk Tiers**")
    st.markdown("🟢 **CLEAR** — below low-FRR threshold")
    st.markdown("🟡 **REVIEW** — uncertain, needs analyst")
    st.markdown("🔴 **ESCALATE** — above low-FAR threshold")
    st.divider()
    st.markdown("**Model:** One-Class SVM")
    st.markdown("**Features:** 7 keystroke timing")
    st.markdown("**Enrollment:** 10 legitimate sessions")

# Reset state if user changes
if selected_user != st.session_state.last_user:
    st.session_state.session_idx    = 0
    st.session_state.review_queue   = []
    st.session_state.current_session = None
    st.session_state.metrics = {
        'processed': 0, 'clear': 0, 'review': 0,
        'escalate': 0, 'confirmed_fraud': 0, 'false_alarm': 0
    }
    st.session_state.last_user = selected_user

# Train model
with st.spinner(f"Building behavioral profile for user {selected_user}..."):
    model, low_frr_thresh, low_far_thresh, centroid = train_model(selected_user)

# All sessions for selected user
user_sessions = df[df['user_id'] == selected_user].reset_index(drop=True)

# Pre-compute score normalization bounds
all_raw_scores = -model.decision_function(user_sessions[KEYSTROKE_FEATURES].values)
score_min = all_raw_scores.min()
score_max = all_raw_scores.max()
score_range = score_max - score_min + 1e-9

norm_low_frr = (low_frr_thresh - score_min) / score_range
norm_low_far = (low_far_thresh - score_min) / score_range

# ── Main layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📡 Incoming Session")

    if st.button("▶ Run Next Session", type="primary", use_container_width=True):
        if st.session_state.session_idx < len(user_sessions):
            session  = user_sessions.iloc[st.session_state.session_idx]
            features = session[KEYSTROKE_FEATURES].values.reshape(1, -1)
            raw_score = float(-model.decision_function(features))
            norm_score = (raw_score - score_min) / score_range

            tier, color, icon = get_risk_tier(norm_score, norm_low_frr, norm_low_far)

            st.session_state.current_session = {
                'session_id': session['session_id'],
                'label':      session['label'],
                'features':   session[KEYSTROKE_FEATURES].values,
                'score':      norm_score,
                'raw_score':  raw_score,
                'tier':       tier,
                'color':      color,
                'icon':       icon,
                'idx':        st.session_state.session_idx
            }

            st.session_state.session_idx += 1
            st.session_state.metrics['processed'] += 1

            if tier == 'CLEAR':
                st.session_state.metrics['clear'] += 1
            elif tier == 'REVIEW':
                st.session_state.metrics['review'] += 1
                st.session_state.review_queue.append(
                    st.session_state.current_session.copy()
                )
            else:
                st.session_state.metrics['escalate'] += 1
                st.session_state.review_queue.append(
                    st.session_state.current_session.copy()
                )
        else:
            st.warning("All sessions processed for this user.")

    if st.session_state.current_session:
        cs = st.session_state.current_session

        # Risk tier badge
        st.markdown(
            f"<div style='background:{cs['color']};padding:12px;border-radius:8px;"
            f"text-align:center;font-size:22px;font-weight:bold;color:white;'>"
            f"{cs['icon']} {cs['tier']}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Session {cs['session_id']} — Anomaly Score: {cs['score']:.3f}")

        # Score bar
        st.pyplot(score_bar(cs['score'], norm_low_frr, norm_low_far))

        # Deviation chart
        st.pyplot(deviation_chart(cs['features'], centroid))

        # Reveal ground truth
        if st.button("🔍 Reveal Ground Truth"):
            if cs['label'] == 'impostor':
                st.error("🚨 IMPOSTOR — This was a fraudulent session")
            else:
                st.success("✅ LEGITIMATE — This was the real user")


with right_col:
    st.subheader("📋 Review Queue")

    if not st.session_state.review_queue:
        st.info("No sessions flagged for review yet.")
    else:
        for i, queued in enumerate(st.session_state.review_queue):
            with st.expander(
                f"{queued['icon']} Session {queued['session_id']} — "
                f"Score: {queued['score']:.3f} — {queued['tier']}"
            ):
                deviations = queued['features'] - centroid
                feat_df = pd.DataFrame({
                    'Feature':   [FEATURE_LABELS[f] for f in KEYSTROKE_FEATURES],
                    'Deviation': [round(float(d), 4) for d in (np.array(queued['features'], dtype=float) - np.array(centroid, dtype=float))]
                })
                st.dataframe(feat_df, hide_index=True, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚨 Confirm Fraud", key=f"fraud_{i}",
                                 use_container_width=True):
                        st.session_state.metrics['confirmed_fraud'] += 1
                        st.session_state.review_queue.pop(i)
                        st.rerun()
                with col2:
                    if st.button("✅ False Alarm", key=f"false_{i}",
                                 use_container_width=True):
                        st.session_state.metrics['false_alarm'] += 1
                        st.session_state.review_queue.pop(i)
                        st.rerun()


# ── Operations Dashboard ──────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Operations Dashboard")

m  = st.session_state.metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Sessions Processed", m['processed'])
c2.metric("🟢 Clear",           m['clear'])
c3.metric("🟡 Review",          m['review'])
c4.metric("🔴 Escalate",        m['escalate'])
c5.metric("Confirmed Fraud",    m['confirmed_fraud'])
c6.metric("False Alarms",       m['false_alarm'])

if m['processed'] > 0:
    flag_rate = (m['review'] + m['escalate']) / m['processed'] * 100
    st.caption(f"Flag rate: {flag_rate:.1f}% of sessions routed for review")

if m['confirmed_fraud'] + m['false_alarm'] > 0:
    fpr = m['false_alarm'] / (m['false_alarm'] + m['clear'] + 1e-9) * 100
    st.caption(f"False positive rate on reviewed cases: {fpr:.1f}%")