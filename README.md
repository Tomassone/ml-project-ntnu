# Predicting Hydro Unit Commitment with ML · TDT4173

[![Course](https://img.shields.io/badge/course-TDT4173_Machine_Learning-blue)](https://www.ntnu.edu/studies/courses/TDT4173)
[![Partner](https://img.shields.io/badge/partner-SINTEF_Energy_Research-00A651)](https://www.sintef.no/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Repository for our machine learning group's project — a Kaggle-style competition run with
[SINTEF Energy Research](https://www.sintef.no/), part of the **InterOpt** research project on combining
machine learning with optimization.

**Team `[25] Nah, i'd win`** · Francesco Binni · Cesare Tomasi · Cristina Zanetti

---

## 🎯 The task

Short-term hydropower scheduling relies on solving the **Unit Commitment (UC) problem**: deciding whether
each generating unit should be ON or OFF at every hour of a scheduling horizon. Traditional software such as
[SHOP](https://shop-open.sintef.energy/) solves this with expensive mathematical optimization, and a solution
for one case cannot be reused for another.

The goal of this project is to **learn those decisions instead of optimizing them**. The model:

- **takes** time-series inputs — day-ahead electricity price, reservoir inflow, storage level, water value —
  plus optional information such as environmental constraints and watercourse topology;
- **outputs** a probability in `[0, 1]` for every generating unit and every hour, where a value close to 1
  means high confidence the unit is ON.

Accurate predictions can then be used to warm-start and speed up the optimization itself.

## ⚡ The system

The study uses real data from the **Tokke–Vinje hydropower system** in southern Norway:

| | |
|---|---|
| Reservoirs | 17 |
| Power plants | 9 |
| Generating units | 14 |
| Installed capacity | ~1000 MW |
| Annual production | ~4.9 TWh |

Each scheduling case is one week (**168 hours**), produced by a daily rolling-horizon SHOP run.
The dataset covers daily cases from 2015-01-01 to 2024-12-25 — **3,647 cases**: 2,922 for training,
725 to predict.

## 📁 Repository structure

```
ml-project-ntnu/
├── notebooks/
│   ├── Short_notebook_1.ipynb     # Model A: XGBoost + LightGBM + two sequence networks
│   ├── Short_notebook_2.ipynb     # Model B: CatBoost + ExtraTrees + TCN (logit blend)
│   ├── requirements_1.txt         # Exact environment for notebook 1 (Python 3.12)
│   └── requirements_2.txt         # Environment for notebook 2
├── training_data/
│   ├── kernel/                    # Core dataset
│   │   ├── Historical_day_ahead_price_2015_2025.csv
│   │   ├── Historical_inflow_1958_2025.csv
│   │   ├── Historical_volume_2015_2024.csv
│   │   ├── Synthetic_water_value_2015_2024.csv
│   │   └── Unit_commitment_decisions.csv      # Training labels (SHOP solutions)
│   ├── extended/                  # Optional extras
│   │   ├── Constraint_min_flow.csv
│   │   ├── Constraint_min_volume.csv
│   │   └── Tokke_Vinje_topology.yaml          # Reservoir/plant connections
│   ├── prediction_mapping.csv     # The 725 test cases to predict
│   └── sample_submission.csv      # 725 rows × 2,352 unit-hour columns
├── docs/
│   ├── ML_Task_TDT4173.pdf        # Full task description (SINTEF)
│   └── Dataset_definitions_and_explanation.pdf
├── TDT4173_Report.pdf             # Our project report
└── LICENSE
```

## 🧠 Approach

The two notebooks answer the same question for every row of data: *"in this week, is this unit ON at this
hour?"* — one row per **(case, unit, hour)** combination, with engineered features (margins, rolling
statistics, weekly quantiles, topology-aware reservoir groups, …) and a 0/1 target.

To maximize blend diversity, we built **two independent pipelines from scratch** and merged them:

| | Notebook 1 | Notebook 2 |
|---|---|---|
| **Tree models** | XGBoost + LightGBM | CatBoost + ExtraTrees |
| **Neural models** | Sequence network + "refiner" network (fed with out-of-fold XGBoost probabilities) | Dilated-convolution TCN |
| **Ensembling** | Weighted average: `0.20·XGB + 0.30·LGBM + 0.25·NN + 0.25·refiner` | Logit-space blend of the three models |
| **Validation** | Temporal cross-validation, confirmed on 2019 (never used for tuning) | Temporal hold-out on the most recent 15% of cases, **with a 7-day purge** to avoid leakage from overlapping weeks |

Both pipelines self-locate the competition files, use the GPU when available (CPU otherwise), and run the
neural networks over multiple seeds, averaging predictions.

## 🚀 Getting started

Each notebook is self-contained: place it in the same folder as (or one level above) `training_data/` and run
it top to bottom — no paths to edit. It writes a validated `submission.csv` (725 rows × 2,352 columns of
probabilities).

```bash
# Notebook 1
pip install -r notebooks/requirements_1.txt
jupyter execute notebooks/Short_notebook_1.ipynb

# Notebook 2
pip install -r notebooks/requirements_2.txt
jupyter execute notebooks/Short_notebook_2.ipynb
```

> 💡 GPU is optional. Notebook 1 was developed with PyTorch CUDA 13.0 on an 8 GB laptop GPU; without one the
> CPU build works and the notebooks detect the device themselves (fewer NN seeds are used on CPU).

## 📚 References

- **InterOpt** — SINTEF Energy Research project combining machine learning with hydropower optimization.
- [SHOP (Short-term Hydropower Optimization)](https://shop-open.sintef.energy/) — the optimization tool that
  generated the training labels.

## 📄 License

Released under the [MIT License](LICENSE).
