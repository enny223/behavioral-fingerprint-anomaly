# Behavioral Fingerprint Anomaly Detection

Continuous authentication system that detects session imposters using keyboard and mouse behavioral dynamics — without relying on passwords or explicit fraud labels.

---

## Problem

Once a user authenticates, most systems extend trust for the remainder of the session. This project asks a different question: **can we continuously verify that the person currently interacting with a system is the same person who logged in, using only the behavioral signature of how they type and move their mouse?**

This is an unsupervised anomaly detection problem. We have no labeled impostor data at training time. We build a behavioral profile for each user from legitimate sessions only, then score new sessions against that profile — flagging significant deviations as suspicious.

---

## Dataset

**Behaviour Biometrics Dataset** — Edge Hill University / CyberSIgnature Project  
Published on Mendeley Data (CC BY 4.0): https://data.mendeley.com/datasets/fnf8b85kr6/1

- 88 users, 20 sessions each (1,760 total instances)
- Collected via a simulated online card payment form
- Raw signals: key press, key release, mouse movement, mouse press, mouse release
- Ground truth labels (legitimate / impostor) used **only for evaluation**, not training

---

## Approach

1. **Feature Engineering** — extract dwell time, flight time, digraph timing, mouse velocity, trajectory directness, hover time, backspace rate, and session-level aggregates from raw event streams
2. **Per-user Behavioral Profiling** — build one anomaly model per user trained exclusively on their legitimate sessions
3. **Anomaly Scoring** — score held-out sessions and produce a continuous anomaly score
4. **Threshold Analysis** — plot False Rejection Rate vs. False Acceptance Rate to find the operational threshold
5. **Enrollment Sensitivity** — measure how performance degrades as enrollment data is reduced from 10 → 5 → 3 → 1 sessions

---

## Structure

```
behavioral-fingerprint-anomaly/
├── data/               # Raw dataset lives here — not tracked by git
├── notebooks/          # Exploration and analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                # Reusable modules
│   ├── __init__.py
│   ├── features.py     # Feature extraction from raw JSON
│   ├── profiler.py     # Per-user behavioral profile builder
│   └── scoring.py      # Anomaly scoring and threshold logic
├── reports/            # Figures and result summaries
├── outputs/            # Model artifacts — not tracked by git
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/enny223/behavioral-fingerprint-anomaly.git
cd behavioral-fingerprint-anomaly

# Create and activate environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download dataset
# https://data.mendeley.com/datasets/fnf8b85kr6/1
# Place contents inside the data/ folder
```

---

## Key Concepts

- **Dwell Time** — how long a key is held down
- **Flight Time** — gap between releasing one key and pressing the next
- **False Rejection Rate (FRR)** — legitimate user flagged as impostor
- **False Acceptance Rate (FAR)** — impostor accepted as legitimate user
- **Equal Error Rate (EER)** — the threshold where FRR = FAR; primary evaluation metric

---

## References

- Nnamoko et al. (2022). *Behaviour Biometrics Dataset*. Mendeley Data. https://doi.org/10.17632/fnf8b85kr6.1
- Killourhy & Maxion (2009). *Comparing Anomaly-Detection Algorithms for Keystroke Dynamics*. DSN-2009. https://www.cs.cmu.edu/~maxion/pubs/KillourhyMaxion09.pdf
