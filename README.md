<p align="center">
  <img src="assets/banner.png" alt="BioSignals-Stress-Modeling banner" width="100%">
</p>

<h1 align="center">BioSignals-Stress-Modeling</h1>

<p align="center">
  An end-to-end pipeline linking wearable physiological signals to circadian rhythms, stress, and personality traits.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status">
  <img src="https://img.shields.io/github/stars/MoustafAbdelwahap/BioSignals-Stress-Modeling?style=social" alt="Stars">
</p>

---

## What this is

A research pipeline (originally built during a master's research internship) for turning large-scale, multi-day wearable physiological data into models of stress and personality — going from raw sensor streams all the way to personalized, subject-aware classifiers.

- **Data cleaning** — denoising and aligning multimodal biosignal streams, outlier detection with tunable methods and window lengths
- **Feature extraction** — statistical, frequency-domain, and physiological features from HR, HRV, EDA, skin temperature, and accelerometer data
- **Exploratory analysis** — participant demographics, questionnaire/diary response distributions, data quality and coverage per user
- **Circadian modelling** — recovering 24h rhythm patterns from continuous signals, then testing whether those rhythms differ by psychological trait or stress level
- **Clustering & statistical testing** — grouping participants by psychological profile and testing whether physiology and daily-diary responses differ across clusters
- **Stress & personality prediction** — classifiers/regressors that progressively relax the problem definition (extreme-case binary → moderate binary → 3-class), evaluated under both cross-validation and leave-one-subject-out (LOSO)
- **Model interpretability & fairness** — feature importance, error analysis, and AI ethics/fairness considerations given the sensitive nature of physiological + psychological data

## Why

Wearables generate huge volumes of raw physiological data, but most of it is noise until it's cleaned, contextualized against circadian rhythms, and linked to psychological state. Most published pipelines stop at "one model for everyone." This project treats that as a starting point, not the goal — the roadmap below moves toward personalized, subject-aware modelling instead of one-size-fits-all classifiers.

## Dataset

<!-- Fill in with your actual dataset name/source -->
A multimodal physiological dataset collected from participants over multiple days, combining continuous wearable-sensor streams with:
- **Pre-study questionnaires** — psychological trait / personality measures
- **Daily diaries** — self-reported stress levels and state, answered repeatedly across the study period
- **Continuous biosignals** — recorded throughout, at varying quality depending on sensor placement, movement, and device limitations

| Signal | Description | Typical Use |
|---|---|---|
| Heart Rate | Beat-to-beat cardiac activity | Stress/arousal indicator |
| HRV | Heart rate variability | Autonomic nervous system state |
| EDA | Electrodermal activity | Skin conductance, arousal |
| Skin Temperature | Thermoregulation | Circadian phase marker |
| Accelerometer | Movement | Activity level, sleep/wake detection |

Exploratory analysis covers participant-level statistics (age, sex distribution), questionnaire/diary response distributions, and per-user data coverage (hours of signal per day, number of diary responses, data quality).

> Note: raw data is not included in this repo. See [`data/README.md`](data/README.md) for access instructions.

## Pipeline overview

```
Raw sensor streams + questionnaires + daily diaries
      │
      ▼
 1. Data Cleaning & Outlier Detection   (src/preprocessing.py)
      │
      ▼
 2. Exploratory Data Analysis           (notebooks/02_*)
      │   participant stats · questionnaire/diary distributions · data quality
      ▼
 3. Feature Extraction                  (src/features.py)
      │
      ▼
 4. Circadian Modelling                 (src/circadian.py)
      │   24h rhythm recovery · rhythm vs. trait / stress-level comparison
      ▼
 5. Clustering & Statistical Testing    (notebooks/05_*)
      │   psycho-profile clusters · diary/physiology differences across clusters
      ▼
 6. Stress & Personality Classifiers    (src/models.py)
      │   binary (extreme) → binary (moderate) → 3-class
      │   feature selection · normalization strategies
      │   evaluation: cross-validation vs. leave-one-subject-out (LOSO)
      ▼
 7. Interpretability, Fairness & Personalization
          feature importance · error analysis · bias auditing · multi-task / per-subject models
```

## Methodology notes

- **Problem definition is relaxed progressively**: stress classification starts with extreme cases only (e.g. diary scores `<30` vs `>70`), then tightens toward the middle (`<40` vs `>60`), then expands to 3 classes (low / neutral / high) as models improve.
- **Two evaluation regimes are reported side by side**: standard cross-validation and leave-one-subject-out (LOSO). LOSO scores are expected to be lower — that gap is itself a diagnostic for how much a model is relying on subject-specific signal rather than generalizable patterns.
- **Personality is treated in both directions**: clustering participants by psychological trait to see if physiology/diary responses differ across clusters, and — going further — testing whether personality itself can be predicted from physiology.
- **Fairness is treated as a first-class concern**, not an afterthought, given the sensitive nature of physiological + psychological data.

## Results

<!-- Replace with your actual headline numbers/plots -->
| Model | Task | Evaluation | Metric | Score |
|---|---|---|---|---|
| Baseline (Logistic Regression) | Stress, extreme binary (<30 vs >70) | Cross-validation | F1 | 0.XX |
| Baseline (Logistic Regression) | Stress, extreme binary (<30 vs >70) | LOSO | F1 | 0.XX |
| Random Forest | Stress, 3-class | Cross-validation | F1 (macro) | 0.XX |
| Random Forest | Stress, 3-class | LOSO | F1 (macro) | 0.XX |
| Clustering | Personality trait grouping | — | Silhouette | 0.XX |

<p align="center">
  <img src="assets/figures/circadian_pattern_example.png" width="70%" alt="Example circadian pattern">
</p>

## Repository structure

```
BioSignals-Stress-Modeling/
├── notebooks/          # step-by-step exploratory + modelling notebooks
├── src/                # reusable pipeline code (cleaning, features, models)
├── assets/             # figures used in this README
├── data/               # dataset access instructions (data itself is gitignored)
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/MoustafAbdelwahap/BioSignals-Stress-Modeling.git
cd BioSignals-Stress-Modeling
pip install -r requirements.txt
```

Run the pipeline stage by stage via the notebooks in `notebooks/`, or import individual modules from `src/`:

```python
from src.preprocessing import clean_signals
from src.features import extract_features
from src.circadian import fit_circadian_model

df_clean = clean_signals(raw_df)
features = extract_features(df_clean)
circadian_fit = fit_circadian_model(features)
```

## Roadmap

- [ ] Explore raw-signal-as-input models (RNN/CNN) instead of hand-crafted features
- [ ] Move from one-size-fits-all classifiers toward **personalized, multi-task models** — treating each subject (or subject group) as a task rather than pooling everyone into one model
- [ ] Systematic bias/fairness auditing of trained models
- [ ] Package feature extraction as a standalone pip-installable module
- [ ] Post-processing and deployment path for real-world / real-time use
- [ ] Add unit tests for `src/`

## Related reading

This pipeline's design was informed by the following surveys on deep learning for physiological signals:

- Deep Learning in Physiological Signal Data: A Survey
- Deep Learning on 1-D Biosignals: a Taxonomy-based Survey
- Deep learning for healthcare applications based on physiological signals: A review



Dataset Overview
The project uses a large multimodal physiological dataset collected from participants over multiple days. It combines continuous wearable‑sensor signals with self‑reported psychological data, enabling both temporal modelling and human‑state prediction.

Physiological Signals
Each participant contributes several streams of biosignals, typically sampled at high frequency:

Heart Rate — beat‑to‑beat cardiac activity

HRV (Heart Rate Variability) — autonomic nervous system indicator

EDA (Electrodermal Activity) — stress/arousal‑related skin conductance

Skin Temperature — thermoregulation and circadian cues

Accelerometer — movement, activity level, sleep/wake patterns

Signals are recorded continuously, producing hours of data per day per participant, with varying quality depending on sensor placement, movement, and device limitations.
