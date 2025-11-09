#!/usr/bin/env python3

"""
Run QNLP (lambeq) embeddings + PCA + QMCMC (QD-HMC) for binary genomics tasks.

Steps:
1) Load lambeq embeddings (train/val/test) from --embeddings_dir
2) Fit StandardScaler + PCA on train, transform all splits
3) Define Bayesian logistic posterior (energy + gradient)
4) Sample parameters using QD-HMC
5) Compute posterior predictive metrics and save artifacts to --output_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

from qmcmc.data import load_embeddings
from qmcmc.features import build_feature_pipeline, save_preprocessing
from qmcmc.posteriors import logistic_energy_and_grad, predict_proba
from qmcmc.qdhmc_adapter import QDHMCSampler
from qmcmc.evaluate import (
    posterior_predictive,
    compute_metrics,
    compute_detailed_metrics,
    save_json,
    save_numpy,
)
from qmcmc.plots import (
    plot_roc_curves,
    plot_confusion_matrices,
    plot_energy_trace,
    plot_posterior_traces,
    plot_posterior_pairplot,
    plot_decision_boundary,
    plot_metric_bars,
    plot_classification_report_table,
)


def main():
    parser = argparse.ArgumentParser(description="QNLP + QMCMC (QD-HMC) runner")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "lambeq_embeddings"),
        help="Directory with lambeq embeddings train.pt|val.pt|test.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Output directory for metrics and artifacts",
    )
    parser.add_argument("--pca_dims", type=int, default=2, help="PCA components (2–3 recommended)")
    parser.add_argument("--precision", type=int, default=2, help="Qubits per parameter (<=3)")
    parser.add_argument("--t", type=float, default=0.5, help="Total evolution time")
    parser.add_argument("--r", type=int, default=2, help="Trotter steps")
    parser.add_argument("--eta_mu", type=float, default=0.5, help="Mean of eta (kinetic) parameter")
    parser.add_argument("--eta_sigma", type=float, default=0.4, help="Std of eta (kinetic) parameter")
    parser.add_argument("--lambda_mu", type=float, default=0.5, help="Mean of lambda (potential) parameter")
    parser.add_argument("--lambda_sigma", type=float, default=0.4, help="Std of lambda (potential) parameter")
    parser.add_argument("--burnin", type=int, default=200, help="Burn-in samples")
    parser.add_argument("--samples", type=int, default=500, help="Sampling draws after burn-in")
    parser.add_argument("--progress", action="store_true", help="Show sampling progress (tqdm if available)")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N iters when tqdm unavailable")
    parser.add_argument("--prior_std", type=float, default=1.0, help="Gaussian prior std for parameters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--allow_single_class",
        action="store_true",
        help=(
            "Proceed even if a split has a single class. "
            "By default the script fails fast when class labels are degenerate "
            "(e.g., all ones), because metrics become meaningless."
        ),
    )

    parser.set_defaults(progress=True)
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Basic argument validation to avoid runtime errors
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.burnin < 0:
        raise ValueError("--burnin must be >= 0")
    if args.pca_dims <= 0:
        raise ValueError("--pca_dims must be > 0")

    # Resolve embeddings directory; if missing, try simple fallback or auto-generate
    def _has_required_embeddings(d: Path) -> bool:
        return all((d / f).exists() for f in ("train.pt", "val.pt", "test.pt"))

    def _ensure_embeddings(user_dir: Path) -> Path:
        # 1) Use provided dir if valid
        if user_dir and user_dir.exists() and _has_required_embeddings(user_dir):
            print(f"[Data] Using embeddings at: {user_dir}")
            return user_dir

        # 2) Try sibling lambeq_embeddings
        proj_dir = Path(__file__).resolve().parents[1]
        lambeq_dir = proj_dir / "lambeq_embeddings"
        if lambeq_dir.exists() and _has_required_embeddings(lambeq_dir):
            print(f"[Data] Using lambeq embeddings at: {lambeq_dir}")
            return lambeq_dir

        # 3) Try sibling simple_embeddings
        simple_dir = proj_dir / "simple_embeddings"
        if simple_dir.exists() and _has_required_embeddings(simple_dir):
            print(f"[Data] Using simple embeddings at: {simple_dir}")
            return simple_dir

        # 4) Generate simple embeddings from processed_data
        processed = proj_dir / "processed_data"
        if not processed.exists():
            raise FileNotFoundError(
                f"Processed data not found at {processed}. Run preprocess_genomics.py first "
                f"or pass --embeddings_dir to an existing directory."
            )

        print("[Data] No embeddings found. Generating simple embeddings from processed_data...")
        if str(proj_dir) not in sys.path:
            sys.path.insert(0, str(proj_dir))
        try:
            from lambeq_encoder import SimpleKmerEncoder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import SimpleKmerEncoder from lambeq_encoder.py. "
                "Ensure it exists and is importable."
            ) from e

        gen_dir = simple_dir
        encoder = SimpleKmerEncoder(
            data_dir=str(processed),
            output_dir=str(gen_dir),
            embedding_dim=128,
            seed=42,
        )
        encoder.run()
        if not _has_required_embeddings(gen_dir):
            raise RuntimeError(
                f"Embedding generation did not create expected files in {gen_dir}."
            )
        print(f"[Data] Simple embeddings created at: {gen_dir}")
        return gen_dir

    resolved_dir = _ensure_embeddings(Path(args.embeddings_dir))
    embeddings = load_embeddings(resolved_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print class balance for sanity
    def _balance(y):
        y = np.asarray(y).astype(int)
        total = len(y)
        pos = int(np.sum(y))
        return total, pos, total - pos
    tr_tot, tr_pos, tr_neg = _balance(embeddings.train.y)
    va_tot, va_pos, va_neg = _balance(embeddings.val.y)
    te_tot, te_pos, te_neg = _balance(embeddings.test.y)
    print(f"[Data] Train balance: total={tr_tot}, pos={tr_pos}, neg={tr_neg}")
    print(f"[Data] Val balance:   total={va_tot}, pos={va_pos}, neg={va_neg}")
    print(f"[Data] Test balance:  total={te_tot}, pos={te_pos}, neg={te_neg}")

    # Fail fast on degenerate labels unless explicitly allowed
    def _has_two_classes(y: np.ndarray) -> bool:
        return np.unique(np.asarray(y).astype(int)).size >= 2

    bad_splits = [
        name
        for name, y in (
            ("train", embeddings.train.y),
            ("val", embeddings.val.y),
            ("test", embeddings.test.y),
        )
        if not _has_two_classes(y)
    ]
    if bad_splits and not args.allow_single_class:
        msg = (
            "One or more splits are single-class ({}). "
            "Results like 100% accuracy with undefined ROC-AUC are expected in this case and are not meaningful.\n"
            "Please regenerate data with both classes present (see preprocess_genomics.py) "
            "or pass --allow_single_class to proceed for debugging only."
        ).format(", ".join(bad_splits))
        raise RuntimeError(msg)

    # Fit feature pipeline on train and transform splits
    n_features = embeddings.train.X.shape[1]
    if args.pca_dims > min(n_features, embeddings.train.X.shape[0]):
        raise ValueError(
            f"--pca_dims={args.pca_dims} exceeds allowable limit min(n_features, n_samples)="
            f"{min(n_features, embeddings.train.X.shape[0])}"
        )
    pipe = build_feature_pipeline(embeddings.train.X, n_components=args.pca_dims)
    Z_train = pipe.transform(embeddings.train.X)
    Z_val = pipe.transform(embeddings.val.X)
    Z_test = pipe.transform(embeddings.test.X)

    # Save preprocessing info
    save_preprocessing(pipe, out_dir / "preprocessing.json")

    # Define posterior energy + gradient closures on train data
    def energy_fn(theta: np.ndarray) -> float:
        e, _ = logistic_energy_and_grad(theta, Z_train, embeddings.train.y, prior_std=args.prior_std)
        return float(e)

    def grad_fn(theta: np.ndarray) -> np.ndarray:
        _, g = logistic_energy_and_grad(theta, Z_train, embeddings.train.y, prior_std=args.prior_std)
        return g

    d = Z_train.shape[1]
    num_vars = d + 1  # weights + bias

    # Build QMCMC sampler (QD-HMC)
    sampler = QDHMCSampler(
        energy_fn=energy_fn,
        grad_fn=grad_fn,
        num_vars=num_vars,
        precision=args.precision,
        t=args.t,
        r=args.r,
        eta_mu=args.eta_mu,
        eta_sigma=args.eta_sigma,
        lambda_mu=args.lambda_mu,
        lambda_sigma=args.lambda_sigma,
    )

    # Sample posterior parameters
    print("\n[QMCMC] Sampling posterior parameters with QD-HMC...")
    samples, acc_rate, diagnostics = sampler.sample(
        burnin=args.burnin,
        samples=args.samples,
        init=np.zeros(num_vars, dtype=np.float64),
        seed=args.seed,
        progress=args.progress,
        log_every=args.log_every,
        save_circuit_path=str(out_dir / "qmcmc_circuit.png"),
    )
    print(f"[QMCMC] Acceptance rate: {acc_rate:.1%}")

    # Posterior predictive on splits
    print("[Eval] Computing posterior predictive...")
    mean_tr, std_tr = posterior_predictive(samples, Z_train, predict_proba)
    mean_va, std_va = posterior_predictive(samples, Z_val, predict_proba)
    mean_te, std_te = posterior_predictive(samples, Z_test, predict_proba)

    # Metrics
    metrics = {
        "train": compute_detailed_metrics(embeddings.train.y, mean_tr),
        "val": compute_detailed_metrics(embeddings.val.y, mean_va),
        "test": compute_detailed_metrics(embeddings.test.y, mean_te),
    }

    # Persist artifacts
    save_json(metrics, out_dir / "metrics.json")
    # Save a file that mirrors run_genomics_training.py naming for quick comparison
    training_like = {
        "test_acc": metrics["test"]["accuracy"] * 100.0,
        "test_f1": metrics["test"]["f1_weighted"],
        "test_loss": metrics["test"]["loss"],
        "confusion_matrix": metrics["test"]["confusion_matrix"],
        "classification_report": metrics["test"]["classification_report"],
        "val_acc": metrics["val"]["accuracy"] * 100.0,
        "train_acc": metrics["train"]["accuracy"] * 100.0,
    }
    save_json(training_like, out_dir / "results_training_like.json")
    diag_payload = {
        "acceptance_rate": acc_rate,
        "n_samples": int(args.samples),
        "class_balance": {
            "train": {"total": tr_tot, "pos": tr_pos, "neg": tr_neg},
            "val": {"total": va_tot, "pos": va_pos, "neg": va_neg},
            "test": {"total": te_tot, "pos": te_pos, "neg": te_neg},
        },
        "single_class_splits": bad_splits,
        "allow_single_class": bool(args.allow_single_class),
    }
    save_json(diag_payload, out_dir / "diagnostics.json")
    save_numpy(samples, out_dir / "posterior_samples.npy")

    # Figures
    print("[Plots] Generating figures...")
    y_by_split = {"train": embeddings.train.y, "val": embeddings.val.y, "test": embeddings.test.y}
    p_by_split = {"train": mean_tr, "val": mean_va, "test": mean_te}
    plot_roc_curves(y_by_split, p_by_split, out_dir / "roc_curves.png")
    plot_confusion_matrices(y_by_split, p_by_split, out_dir / "confusion_matrices.png")

    # Diagnostics
    energies = np.asarray(diagnostics.get("energies", []), dtype=np.float64)
    if energies.size > 0:
        plot_energy_trace(energies, out_dir / "energy_trace.png")

    # Posterior summaries
    plot_posterior_traces(samples, out_dir / "posterior_traces.png")
    plot_posterior_pairplot(samples, out_dir / "posterior_pairplot.png", max_dims=4)

    # Decision boundary (2D PCA only)
    if Z_train.shape[1] == 2:
        theta_mean = samples.mean(axis=0)
        plot_decision_boundary(Z_train, embeddings.train.y, theta_mean, out_dir / "decision_boundary.png")

    # Summary metric figures aligned with training script
    plot_metric_bars(metrics, out_dir / "summary_metrics.png")
    cr_test = metrics["test"].get("classification_report")
    if cr_test:
        plot_classification_report_table(cr_test, out_dir / "classification_report_test.png")

    print("\nDone. Artifacts:")
    print(f"  - {out_dir / 'metrics.json'}")
    print(f"  - {out_dir / 'preprocessing.json'}")
    print(f"  - {out_dir / 'diagnostics.json'}")
    print(f"  - {out_dir / 'posterior_samples.npy'}")
    print(f"  - {out_dir / 'roc_curves.png'}")
    print(f"  - {out_dir / 'confusion_matrices.png'}")
    if (out_dir / 'energy_trace.png').exists():
        print(f"  - {out_dir / 'energy_trace.png'}")
    print(f"  - {out_dir / 'posterior_traces.png'}")
    print(f"  - {out_dir / 'posterior_pairplot.png'}")
    if (out_dir / 'decision_boundary.png').exists():
        print(f"  - {out_dir / 'decision_boundary.png'}")
    print(f"  - {out_dir / 'summary_metrics.png'}")
    if (out_dir / 'classification_report_test.png').exists():
        print(f"  - {out_dir / 'classification_report_test.png'}")


if __name__ == "__main__":
    main()
