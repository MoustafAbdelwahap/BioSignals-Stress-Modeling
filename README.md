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


## Repository structure

```
BioSignals-Stress-Modeling/
├── python/          # step-by-step exploratory + modelling notebooks
├── assets/             # figures used in this README
├── data/               # dataset access instructions (data itself is gitignored)
└── README.md
```
