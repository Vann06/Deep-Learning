# Deep-Learning
Deep Learning y Sistemas Inteligentes
# Proyecto 2 — Deep Learning / AML Detection

---


# 1. Estructura 

```text
Proyecto2/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_data_engineering.ipynb
│   ├── 02_stage_a_autoencoder.ipynb
│   └── 03_stage_b_classifier.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── sequences.py
│   │   ├── splits.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── encoder.py
│   │   ├── autoencoder.py
│   │   ├── attention.py
│   │   └── classifier.py
│   │
│   └── evaluation/
│       ├── anomaly.py
│       └── metrics.py
│
├── artifacts/
│   ├── metadata.json
│   ├── feature_names.json
│   ├── scaler.pkl
│   └── ...
│
├── app/
│   └── app.py
│
├── report/
│   └── sections/
│
└── docs/
    ├── decisions.md
    └── ai_usage.md
```

---
