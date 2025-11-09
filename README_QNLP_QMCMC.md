QNLP + QMCMC Pipeline
=====================

This folder provides a complete example that mixes QNLP embeddings (from
the existing lambeq pipeline) with a quantum–assisted MCMC sampler based on
the True QD‑HMC implementation in `QDHMCMAINFILE.py`.

Overview
--------
- Load lambeq embeddings (`lambeq_embeddings/train.pt|val.pt|test.pt`).
- Standardize and reduce dimensionality (PCA) to 2–3 features.
- Define a Bayesian logistic regression posterior on the reduced features.
- Sample parameters with QD‑HMC (QMCMC) and evaluate predictive performance.

Dependencies (Qiskit required)
-------------------------------
- Python 3.10+
- PyTorch (`torch`) for reading `.pt` files
- NumPy, pandas, matplotlib
- scikit‑learn (PCA, metrics)
- Qiskit (quantum simulation for QD‑HMC)

Install example (if using the repo README env):
```
pip install qiskit numpy pandas matplotlib scikit-learn torch
```

Usage
-----
Run the end‑to‑end script from the repository root:
```
python bradford_hackathon_gene_sequencing/QNLP+QMCMC/run_qnlp_qmcmc.py \
  --embeddings_dir bradford_hackathon_gene_sequencing/lambeq_embeddings \
  --output_dir bradford_hackathon_gene_sequencing/QNLP+QMCMC/results \
  --pca_dims 2 --precision 2 --burnin 200 --samples 500 \
  --t 1.0 --r 3 --eta_mu 0.5 --eta_sigma 0.4 --lambda_mu 0.5 --lambda_sigma 0.4
```

Sanity checks
-------------
- The runner now fails fast if any split is single‑class (e.g., all labels are
  `1`). This prevents misleading metrics like 100% accuracy with undefined ROC AUC.
  If you are intentionally debugging on degenerate labels, add `--allow_single_class`.
- If you hit the check unexpectedly, regenerate balanced data via
  `preprocess_genomics.py` and re‑encode embeddings (see repository root README).

Notes
-----
- The QD‑HMC gate construction scales exponentially in precision. Keep
  `precision <= 3` qubits per parameter (2 is recommended) and use 2–3 PCA
  components so the total qubit count remains tractable for simulation.
- The sampler internally centers near zero; the discrete domain provided
  by the position operator suffices for a toy Bayesian classifier demo.

Outputs
-------
Artifacts are written under `--output_dir`:
- `metrics.json` — accuracy, F1, ROC AUC for train/val/test
- `preprocessing.json` — scaler stats and PCA explained variance
- `posterior_samples.npy` — sampled parameter vectors (num_samples × (d+1))
- `diagnostics.json` — acceptance rate and energy trace summary
- `results_training_like.json` — test_acc/test_f1/test_loss/CM/report (to compare with run_genomics_training.py)
- `roc_curves.png` — ROC curves for train/val/test
- `confusion_matrices.png` — confusion matrices at 0.5 threshold
- `energy_trace.png` — energy (negative log posterior) over samples
- `posterior_traces.png` — trace plots per parameter
- `posterior_pairplot.png` — pairwise scatter/hist of parameters
- `decision_boundary.png` — 2D PCA decision boundary (if `--pca_dims 2`)
- `summary_metrics.png` — bar charts for accuracy, F1-weighted, loss across splits
- `classification_report_test.png` — classification report as a table (test split)

Troubleshooting
---------------
- If Qiskit is missing, install it and try again.
- If lambeq embeddings are not present yet, run `lambeq_encoder.py` first
  (see repository README) or switch `--embeddings_dir` to an existing set.
