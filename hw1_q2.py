# ==============================
# EECE5644 – Assignment 1 – Q2
# 4-class 2D Gaussian classification
# Part A: 0–1 loss (MAP)
# Part B: Unequal loss ERM with given Λ
# ==============================

# ---- imports ----
import numpy as np                          # arrays & math
import matplotlib.pyplot as plt              # plotting (no seaborn)
from numpy.linalg import slogdet, solve      # stable linear algebra

# ---- reproducibility ----
rng = np.random.default_rng(42)              # fixed seed

# ---- helper: multivariate-normal log-pdf (vectorized) ----
def logpdf_mvn(X, mean, cov):
    """Return log N(x | mean, cov) for each row of X (stable)."""
    X = np.atleast_2d(X)                     # ensure 2D (n, d)
    mean = np.asarray(mean).reshape(1, -1)   # row vector (1, d)
    cov = np.asarray(cov)                    # (d, d)
    n, d = X.shape                           # sizes
    XC = X - mean                            # center data
    sign, logdet = slogdet(cov)              # stable log|Σ|
    if sign <= 0:
        raise ValueError("Covariance must be positive definite.")
    inv_times = solve(cov, np.eye(d))        # Σ^{-1} via solve
    quad = np.einsum("ij,ij->i", XC @ inv_times, XC)  # diag(XC Σ^{-1} XC^T)
    return -0.5 * (d * np.log(2*np.pi) + logdet + quad)

# ---- configuration: 4 Gaussians in 2D ----
# Priors (equal): note—labels are 1..4 per problem statement
priors = np.array([0.25, 0.25, 0.25, 0.25])  # P(L=1..4)

# Means (choose your own). Make class 4 the overlapped one (center).
m = {
    1: np.array([-3.0,  0.0]),
    2: np.array([ 3.0,  0.0]),
    3: np.array([ 0.0,  3.0]),
    4: np.array([ 0.0,  0.0]),  # center -> overlaps others
}

# Covariances (choose your own). Give class 4 larger spread.
C = {
    1: np.array([[1.0,  0.2],
                 [0.2,  1.0]]),
    2: np.array([[1.0, -0.2],
                 [-0.2,  1.0]]),
    3: np.array([[1.0,  0.3],
                 [0.3,  1.2]]),
    4: np.array([[1.8,  0.0],  # larger variance -> more overlap
                 [0.0,  1.8]]),
}

# ---- Part A: data generation (10k samples) ----
N = 10_000                                    # total samples
labels = rng.choice([1,2,3,4], size=N, p=priors)  # sample class labels by priors
X = np.empty((N, 2))                           # allocate data array

# draw x ~ N(m_l, C_l) for each selected label l
for l in (1,2,3,4):
    idx = np.where(labels == l)[0]             # indices of class l
    X[idx] = rng.multivariate_normal(m[l], C[l], size=len(idx))

# ---- utility: confusion matrix P(D=i | L=j) ----
def confusion_PD_given_L(pred, truth, num_classes=4):
    """Return confusion matrix whose (i,j) = P(D=i | L=j)."""
    K = num_classes
    M = np.zeros((K, K), dtype=float)         # rows: decisions i; cols: truths j
    for j in range(1, K+1):
        col_idx = (truth == j)
        denom = np.sum(col_idx)
        if denom == 0:
            continue
        for i in range(1, K+1):
            M[i-1, j-1] = np.sum((pred == i) & col_idx) / denom
    return M

# ---- Part A: MAP (0–1 loss) classifier ----
# With equal priors, MAP = pick class with max likelihood p(x|l).
# We'll use log-likelihoods for numerical stability.
loglikes = np.column_stack([
    logpdf_mvn(X, m[1], C[1]),
    logpdf_mvn(X, m[2], C[2]),
    logpdf_mvn(X, m[3], C[3]),
    logpdf_mvn(X, m[4], C[4]),
])                                             # shape (N,4)

# add log-priors (equal here, so this is optional)
logpriors = np.log(priors)[None, :]            # shape (1,4)
logpost_unnorm = loglikes + logpriors          # proportional to posteriors
D_map = 1 + np.argmax(logpost_unnorm, axis=1)  # decisions in {1,2,3,4}

# confusion matrix and probability of error
CM_A = confusion_PD_given_L(D_map, labels)
p_error_A = 1.0 - np.mean(D_map == labels)

print("=== Part A (0–1 loss / MAP) ===")
print("Confusion matrix  P(D=i | L=j):")
print(np.array_str(CM_A, precision=3, suppress_small=True))
print(f"Overall probability of error (empirical): {p_error_A:.4f}\n")

# ---- Part A: scatter plot (different shape per true class; green=correct, red=incorrect) ----
marker_for = {1: ".", 2: "o", 3: "^", 4: "s"}  # dot, circle, triangle, square
plt.figure(figsize=(7,7))
for l in (1,2,3,4):
    idx = np.where(labels == l)[0]
    correct = idx[D_map[idx] == labels[idx]]
    wrong   = idx[D_map[idx] != labels[idx]]
    # correct in green
    if len(correct):
        plt.scatter(X[correct,0], X[correct,1], marker=marker_for[l],
                    s=20, label=f"class {l} (correct)", c="green", alpha=0.6)
    # incorrect in red
    if len(wrong):
        plt.scatter(X[wrong,0], X[wrong,1], marker=marker_for[l],
                    s=20, label=f"class {l} (incorrect)", c="red", alpha=0.6)
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Q2 – Part A: MAP (0–1 loss)\nGreen=correct, Red=incorrect; different marker per TRUE class")
plt.legend(loc="upper right", fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("q2_partA_scatter.png", dpi=160)
plt.show()

# ---- Part B: unequal loss ERM with the given Lambda ----
# rows = decisions i, cols = true labels j  (Λ_{i|j})
Lambda = np.array([
    [  0, 10, 10, 100],
    [  1,  0, 10, 100],
    [  1,  1,  0, 100],
    [  1,  1,  1,   0],
], dtype=float)

# Compute posteriors P(L=j | x) for each sample using Bayes rule:
# P(L=j|x) ∝ p(x|j) P(L=j), and we normalize across j.
# We already have loglikes and logpriors; convert to probabilities safely.
logpost = loglikes + logpriors                       # unnormalized log-posteriors
# subtract max per row before exponentiating (softmax trick)
logpost_shift = logpost - np.max(logpost, axis=1, keepdims=True)
post = np.exp(logpost_shift)
post /= np.sum(post, axis=1, keepdims=True)          # each row sums to 1 (N,4)

# Conditional risk for each decision i: R(i|x) = sum_j Λ_{i|j} * P(L=j|x)
# Build decisions by argmin over i.
# (Vectorized: risks = post @ Λ^T, but careful with orientation.)
risks = post @ Lambda.T                               # (N,4) each column is risk for decision i
D_erm = 1 + np.argmin(risks, axis=1)                 # decisions in {1,2,3,4}

# Confusion matrix P(D=i | L=j) and average risk
CM_B = confusion_PD_given_L(D_erm, labels)
avg_risk = np.mean([Lambda[d-1, l-1] for d, l in zip(D_erm, labels)])

print("=== Part B (Unequal loss ERM) ===")
print("Loss matrix Λ (rows=decision i, cols=true j):")
print(Lambda.astype(int))
print("\nConfusion matrix  P(D=i | L=j):")
print(np.array_str(CM_B, precision=3, suppress_small=True))
print(f"Sample average risk (empirical): {avg_risk:.4f}\n")

# ---- Part B: scatter plot (same visual code) ----
plt.figure(figsize=(7,7))
for l in (1,2,3,4):
    idx = np.where(labels == l)[0]
    correct = idx[D_erm[idx] == labels[idx]]
    wrong   = idx[D_erm[idx] != labels[idx]]
    if len(correct):
        plt.scatter(X[correct,0], X[correct,1], marker=marker_for[l],
                    s=20, label=f"class {l} (correct)", c="green", alpha=0.6)
    if len(wrong):
        plt.scatter(X[wrong,0], X[wrong,1], marker=marker_for[l],
                    s=20, label=f"class {l} (incorrect)", c="red", alpha=0.6)
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Q2 – Part B: ERM with loss matrix Λ\nGreen=correct, Red=incorrect; different marker per TRUE class")
plt.legend(loc="upper right", fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig("q2_partB_scatter.png", dpi=160)
plt.show()
