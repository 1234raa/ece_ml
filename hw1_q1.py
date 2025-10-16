#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import slogdet, solve


# -----------------------------
# Utilities
# -----------------------------
def ensure_outdir(path: str) -> str:
    """Create output directory if missing and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUC with NumPy's trapezoid (or trapz fallback for older NumPy)."""
    order = np.argsort(fpr)
    x = fpr[order]
    y = tpr[order]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))  # NumPy >= 2.0
    return float(np.trapz(y, x))          # Fallback for older NumPy


def logpdf_mvn(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Vectorized log N(x | mean, cov) for rows of X, using stable logdet and solve."""
    X = np.atleast_2d(X)
    mean = np.asarray(mean).reshape(1, -1)
    cov = np.asarray(cov)
    n, d = X.shape
    XC = X - mean
    sign, logdet = slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance must be positive definite.")
    inv_times = solve(cov, np.eye(d))
    quad = np.einsum("ij,ij->i", XC @ inv_times, XC)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def roc_from_scores(scores: np.ndarray, labels: np.ndarray, n_thr: int = 401):
    """Build ROC (FPR, TPR) and error curve by sweeping thresholds across 'scores'."""
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()

    thr = np.concatenate(([-np.inf], np.quantile(scores, np.linspace(0, 1, n_thr)), [np.inf]))
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))

    fpr, tpr, err = [], [], []
    for t in thr:
        pred = (scores > t).astype(int)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        tn = int(np.sum((pred == 0) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        tpr.append(tp / pos if pos else 0.0)
        fpr.append(fp / neg if neg else 0.0)
        err.append((fp + fn) / (pos + neg))
    return np.array(fpr), np.array(tpr), np.array(thr), np.array(err)


def save_roc_figure(fpr: np.ndarray,
                    tpr: np.ndarray,
                    marker_xy=None,
                    title: str = "",
                    out_path: str = "roc.png",
                    marker_label: str = "Min error point"):
    """Plot a single ROC curve and optional marker, then save to out_path."""
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    if marker_xy is not None:
        mx, my = marker_xy
        plt.scatter([mx], [my], marker="o", label=marker_label)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    # Output directory
    OUT = ensure_outdir("outputs")

    # Problem definition
    p0, p1 = 0.65, 0.35
    m0 = np.array([-0.5, -0.5, -0.5])
    m1 = np.array([ 1.0,  1.0,  1.0])
    C0 = np.array([[ 1.0, -0.5,  0.3],
                   [-0.5,  1.0, -0.5],
                   [ 0.3, -0.5,  1.0]])
    C1 = np.array([[ 1.0,  0.3, -0.2],
                   [ 0.3,  1.0,  0.3],
                   [-0.2,  0.3,  1.0]])

    # Dataset generation
    N = 10000
    rng = np.random.default_rng(42)
    N0 = int(round(N * p0))
    N1 = N - N0
    X0 = rng.multivariate_normal(m0, C0, N0)
    X1 = rng.multivariate_normal(m1, C1, N1)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]

    # Save dataset for reproducibility
    dataset_csv = os.path.join(OUT, "dataset.csv")
    np.savetxt(dataset_csv, np.column_stack([X, y]), delimiter=",",
               header="x1,x2,x3,label", comments="")
    
    # -------------------------
    # Part A: ERM (Bayes) with true pdfs
    # -------------------------
    gamma_star = p0 / p1
    log_gamma_star = float(np.log(gamma_star))

    logp0_true = logpdf_mvn(X, m0, C0)
    logp1_true = logpdf_mvn(X, m1, C1)
    scores_A = logp1_true - logp0_true

    fpr_A, tpr_A, thr_A, err_A = roc_from_scores(scores_A, y)
    auc_A = auc_trapezoid(fpr_A, tpr_A)

    # Bayes point at gamma*
    pred_star = (scores_A > log_gamma_star).astype(int)
    tp_star = int(np.sum((pred_star == 1) & (y == 1)))
    fp_star = int(np.sum((pred_star == 1) & (y == 0)))
    fn_star = int(np.sum((pred_star == 0) & (y == 1)))
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    tpr_star = tp_star / pos if pos else 0.0
    fpr_star = fp_star / neg if neg else 0.0
    err_star = (fp_star + fn_star) / N

    # Min-error point (empirical)
    idx_min_A = int(np.argmin(err_A))
    best_thr_A = float(thr_A[idx_min_A])
    best_err_A = float(err_A[idx_min_A])
    best_fpr_A = float(fpr_A[idx_min_A])
    best_tpr_A = float(tpr_A[idx_min_A])

    # Save Part A ROC only
    save_roc_figure(
        fpr_A, tpr_A,
        marker_xy=(best_fpr_A, best_tpr_A),
        title="ROC – Part A: ERM (true Gaussians)",
        out_path=os.path.join(OUT, "roc_partA.png"),
        marker_label="Min error point"
    )

    # -------------------------
    # Part B: Naive Bayes (identity covariance)
    # -------------------------
    I = np.eye(3)
    logp0_nb = logpdf_mvn(X, m0, I)
    logp1_nb = logpdf_mvn(X, m1, I)
    scores_B = logp1_nb - logp0_nb

    fpr_B, tpr_B, thr_B, err_B = roc_from_scores(scores_B, y)
    auc_B = auc_trapezoid(fpr_B, tpr_B)

    idx_min_B = int(np.argmin(err_B))
    best_thr_B = float(thr_B[idx_min_B])
    best_err_B = float(err_B[idx_min_B])
    best_fpr_B = float(fpr_B[idx_min_B])
    best_tpr_B = float(tpr_B[idx_min_B])

    save_roc_figure(
        fpr_B, tpr_B,
        marker_xy=(best_fpr_B, best_tpr_B),
        title="ROC – Part B: Naive Bayes (identity covariance)",
        out_path=os.path.join(OUT, "roc_partB.png"),
        marker_label="Min error point"
    )

    # -------------------------
    # Part C: Fisher LDA (estimated)
    # -------------------------
    X0_s, X1_s = X[y == 0], X[y == 1]  # split by class
    m0_hat, m1_hat = X0_s.mean(axis=0), X1_s.mean(axis=0)
    S0_hat, S1_hat = np.cov(X0_s.T, bias=False), np.cov(X1_s.T, bias=False)
    Sw = S0_hat + S1_hat
    w = solve(Sw, (m1_hat - m0_hat))
    proj = X @ w

    fpr_C, tpr_C, thr_C, err_C = roc_from_scores(proj, y)
    auc_C = auc_trapezoid(fpr_C, tpr_C)

    idx_min_C = int(np.argmin(err_C))
    best_thr_C = float(thr_C[idx_min_C])
    best_err_C = float(err_C[idx_min_C])
    best_fpr_C = float(fpr_C[idx_min_C])
    best_tpr_C = float(tpr_C[idx_min_C])

    save_roc_figure(
        fpr_C, tpr_C,
        marker_xy=(best_fpr_C, best_tpr_C),
        title="ROC – Part C: Fisher LDA",
        out_path=os.path.join(OUT, "roc_partC.png"),
        marker_label="Min error point"
    )

    # -------------------------
    # Combined ROC figure (all three)
    # -------------------------
    plt.figure()
    plt.plot(fpr_A, tpr_A, label="Part A: ERM (true)")
    plt.plot(fpr_B, tpr_B, label="Part B: Naive Bayes (I)")
    plt.plot(fpr_C, tpr_C, label="Part C: Fisher LDA")
    # Markers
    plt.scatter([fpr_star], [tpr_star], marker="o", label="Part A @ γ*")
    plt.scatter([best_fpr_A], [best_tpr_A], marker="x", label="A: min error")
    plt.scatter([best_fpr_B], [best_tpr_B], marker="^", label="B: min error")
    plt.scatter([best_fpr_C], [best_tpr_C], marker="s", label="C: min error")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: ERM vs Naive Bayes vs Fisher LDA")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    roc_all_path = os.path.join(OUT, "roc_all.png")
    plt.savefig(roc_all_path, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Save metrics summary
    # -------------------------
    metrics = {
        "priors": {"p0": p0, "p1": p1},
        "gamma_star": gamma_star,
        "log_gamma_star": log_gamma_star,
        "PartA": {
            "AUC": auc_A,
            "min_error": best_err_A,
            "best_threshold_logLR": best_thr_A,
            "TPR_at_min_error": best_tpr_A,
            "FPR_at_min_error": best_fpr_A,
            "at_gamma_star": {
                "error": err_star,
                "TPR": tpr_star,
                "FPR": fpr_star
            }
        },
        "PartB": {
            "AUC": auc_B,
            "min_error": best_err_B,
            "best_threshold_logLR": best_thr_B,
            "TPR_at_min_error": best_tpr_B,
            "FPR_at_min_error": best_fpr_B
        },
        "PartC": {
            "AUC": auc_C,
            "min_error": best_err_C,
            "best_tau": best_thr_C,
            "TPR_at_min_error": best_tpr_C,
            "FPR_at_min_error": best_fpr_C
        },
        "files": {
            "dataset_csv": dataset_csv,
            "roc_all": roc_all_path,
            "roc_partA": os.path.join(OUT, "roc_partA.png"),
            "roc_partB": os.path.join(OUT, "roc_partB.png"),
            "roc_partC": os.path.join(OUT, "roc_partC.png"),
        }
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------
    # Console summary
    # -------------------------
    print("=== Priors and Bayes threshold ===")
    print(f"p0={p0:.2f}, p1={p1:.2f}, gamma*={gamma_star:.6f}, log(gamma*)={log_gamma_star:.6f}\n")

    print("=== Part A: ERM (true pdfs) ===")
    print(f"AUC={auc_A:.4f}, min error={best_err_A:.4f}, best log-LR thr={best_thr_A:.4f}")
    print(f"At gamma*: error={err_star:.4f}, TPR={tpr_star:.4f}, FPR={fpr_star:.4f}\n")

    print("=== Part B: Naive Bayes (identity covariance) ===")
    print(f"AUC={auc_B:.4f}, min error={best_err_B:.4f}, best log-LR thr={best_thr_B:.4f}\n")

    print("=== Part C: Fisher LDA ===")
    print(f"AUC={auc_C:.4f}, min error={best_err_C:.4f}, best tau={best_thr_C:.4f}\n")

    print("Saved files:")
    for k, v in metrics["files"].items():
        print(f" - {k}: {v}")
    print(f" - metrics: {os.path.join(OUT, 'metrics.json')}")


if __name__ == "__main__":
    main()
