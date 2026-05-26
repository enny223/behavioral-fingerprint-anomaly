# Behavioral Fingerprint Anomaly Detection

Continuous authentication system that detects session impostors using keyboard 
and mouse behavioral dynamics — without relying on passwords or explicit re-authentication.

---

## Problem

Once a user authenticates, most systems extend trust for the remainder of the session.
This project asks: **can we continuously verify that the person currently interacting 
with a system is the same person who logged in, using only the behavioral signature 
of how they type and move their mouse?**

This is an unsupervised anomaly detection problem. We build a behavioral profile for 
each user from legitimate sessions only, then score new sessions against that profile — 
flagging significant deviations as suspicious.

---

## Results

| Model | Mean EER | Median EER | Best | Worst | Users < 50% |
|---|---|---|---|---|---|
| One-Class SVM | **10.85%** | **5.00%** | 0.00% | 60.00% | 86/88 |
| Local Outlier Factor | 15.74% | 15.00% | 0.00% | 45.00% | 88/88 |
| Isolation Forest | 17.27% | 15.00% | 0.00% | 55.00% | 87/88 |

**One-Class SVM is the best performing model.** At the median user, the system correctly 
flags 95% of impostor sessions while accepting 95% of legitimate sessions — trained on 
just 10 sessions and 7 keystroke timing features.

### Enrollment Sensitivity

| Enrollment Sessions | Mean EER | Median EER |
|---|---|---|
| 10 | 6.1% | 0.0% |
| 8 | 14.2% | 5.0% |
| 6 | 14.6% | 7.5% |
| 4 | 18.6% | 13.3% |
| 2 | 22.1% | 16.3% |
| 1 | 25.7% | 21.1% |

**Minimum recommended enrollment: 6 sessions.** Even at 1 session, mean EER of 25.7% 
remains well below random chance (50%).

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

1. **EDA** — structural integrity check, class balance, missingness analysis, 
   raw timing distributions
2. **Feature Engineering** — extract 25 features per session from raw event streams
3. **Modeling** — per-user One-Class SVM, Isolation Forest, and LOF trained on 
   legitimate sessions only
4. **Evaluation** — FRR/FAR curves, EER per user, enrollment sensitivity analysis

### Features Used (7 keystroke timing features)
`dwell_median`, `dwell_mean`, `dwell_std`, `dwell_max`, `flight_median`, 
`flight_min`, `click_dwell_mean`

### Why Unsupervised
- Only 10 legitimate sessions per user — too few for supervised classification
- No impostor labels available at training time in real deployment
- One-class modeling mirrors actual production constraints

---

## Critical Implementation Note

**Score direction inversion** is a subtle but critical pitfall in anomaly detection 
pipelines. All three models expose `decision_function` where higher = more normal. 
`roc_curve` treats higher = more positive (legitimate). Passing anomaly scores 
directly produces EER > 50% — worse than random.

**Fix:** negate scores inside `compute_eer` before passing to `roc_curve`.
See `notebooks/03_modeling.ipynb` for full diagnosis.

---

## Structure
behavioral-fingerprint-anomaly/
├── data/                    # Raw dataset — not tracked by git
├── notebooks/
│   ├── 01_eda.ipynb         # Structural integrity, distributions, missingness
│   ├── 02_feature_engineering.ipynb  # Feature extraction and transformation
│   ├── 03_modeling.ipynb    # Model training and EER evaluation
│   └── 04_evaluation.ipynb  # FRR/FAR curves, enrollment sensitivity
├── src/
│   ├── features.py          # extract_keystroke_features, extract_mouse_features
│   ├── profiler.py          # build_profile, evaluate_user, compute_eer
│   └── scoring.py           # enrollment_sensitivity, optimal_threshold
├── reports/                 # Saved figures
├── environment.yml
└── README.md
---
## Behavioral Session Monitor

A fraud analyst console that simulates how this model operates in a real deployment.
Rather than making hard block/allow decisions, the system routes each session to a
risk tier for human review — mirroring how behavioral biometrics is actually used
in production at companies like BioCatch, Sardine, and NeuroID.

### How it works

Sessions arrive one at a time in mixed order — legitimate and impostor interleaved.
The system scores each session against the enrolled user's behavioral profile and
routes it to one of three tiers:

| Tier | Meaning | Action |
|---|---|---|
| 🟢 CLEAR | Below low-FRR threshold — almost certainly the real user | Pass through, no action |
| 🟡 REVIEW | Uncertain — behavioral signal is ambiguous | Route to analyst queue |
| 🔴 ESCALATE | Above low-FAR threshold — almost certainly an impostor | Freeze session, alert team |

### What the analyst sees

- **Anomaly score bar** — visual indicator of how far the session deviates from normal
- **Feature deviation chart** — which specific behavioral features drove the score
- **Review queue** — flagged sessions with feature breakdown and Confirm/False Alarm buttons
- **Operations dashboard** — live metrics: sessions processed, flag rate, false positive rate

### Why this framing matters

The model's EER of ~11% means it will make mistakes. In a hard block/allow system
that's a problem. In a triage system it's acceptable — the model's job is to surface
suspicious sessions efficiently so human investigators focus their attention where it
matters, not to replace human judgment entirely.

This is how behavioral anomaly detection actually operates at scale.

### Running the app

```bash
# Generate features first
# Run notebooks/01_eda.ipynb and notebooks/02_feature_engineering.ipynb

# Then launch
streamlit run app.py
```

## Setup

```bash
# Clone the repo
git clone https://github.com/enny223/behavioral-fingerprint-anomaly.git
cd behavioral-fingerprint-anomaly

# Create and activate environment
conda env create -f environment.yml
conda activate bfa-env

# Download dataset
# https://data.mendeley.com/datasets/fnf8b85kr6/1
# Extract into data/ folder

# Launch notebooks
jupyter lab
```

---

## References

- Nnamoko et al. (2022). *Behaviour Biometrics Dataset*. Mendeley Data.
  https://doi.org/10.17632/fnf8b85kr6.1
- Killourhy & Maxion (2009). *Comparing Anomaly-Detection Algorithms for Keystroke 
  Dynamics*. DSN-2009. https://www.cs.cmu.edu/~maxion/pubs/KillourhyMaxion09.pdf