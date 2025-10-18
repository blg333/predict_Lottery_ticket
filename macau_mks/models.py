# -*- coding: utf-8 -*-
"""
Predictive models ensemble for 特码 (special number) prediction.
- Frequency balancing
- Omission-based
- Trend (moving window frequency)
- Association via co-occurrence influence
- Clustering (kmeans on embeddings)
- Simple ML (logistic/RF on features)
- ARIMA (statsmodels) on 特码 sequence
- LSTM (Keras) optional
- Bayesian posterior with Dirichlet prior
- Markov chain first-order and two-stage (smoothed)
- Dynamic probability matrix (exp decay on transitions)

All models expose a predict_proba(records) -> np.ndarray[49] over 1..49.
""