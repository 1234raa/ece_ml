import os, io, zipfile, ssl, shutil, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import slogdet, solve

# ============================== CONFIG ==============================
# Choose which datasets to run
RUN_WINE = True
RUN_HAR  = True

# Classifier mode: "QDA" = per-class covariance (default), "LDA" = pooled covariance
MODEL_MODE = "QDA"

# Regularization strength α (λ = α * trace(Σ)/d). Increase if covariances are ill-conditioned.
REG_ALPHA  = 0.05

# Standardize Wine features (z-score). Helps because Wine features are on different scales.
STANDARDIZE_WINE = True

# Which plots to save
PLOT_2D     = True
PLOT_3D     = True
PLOT_BIPLOT = True   # Wine-only 2D PCA biplot (optional)

# Output / data folders
OUTDIR  = "q3_outputs"
DATADIR = "q3_data"
# ====================================================================


# -------------------------- robust downloader --------------------------
def safe_download(url: str, out_path: str):
    """Download URL to out_path using stdlib only.
    Tries verified SSL first; if that fails, retries with unverified SSL (OK for public class datasets)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"[found] {out_path}")
        return
    print(f"[download] {url}")
    try:
        ctx = ssl.create_default_context()  # verified
        with urllib.request.urlopen(url, context=ctx) as r, open(out_path, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"[saved]  {out_path}")
    except Exception as e1:
        print("[warn] verified SSL failed; retrying with unverified context…")
        try:
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ctx) as r, open(out_path, "wb") as f:
                shutil.copyfileobj(r, f)
            print(f"[saved]  {out_path}")
        except Exception as e2:
            raise RuntimeError(
                f"Download failed.\nURL: {url}\nError1: {e1}\nError2: {e2}\n"
                f"(Workaround: download manually to {out_path})"
            )


# ------------------------------ math utils ------------------------------
def logpdf_mvn(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Row-wise log N(x | mean, cov) in a stable way."""
    X = np.atleast_2d(X)
    mean = np.asarray(mean).reshape(1, -1)
    cov  = np.asarray(cov)
    XC = X - mean
    sign, logdet = slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance must be positive definite.")
    inv_cov = solve(cov, np.eye(cov.shape[0]))
    quad = np.einsum("ij,ij->i", XC @ inv_cov, XC)
    return -0.5 * (X.shape[1] * np.log(2*np.pi) + logdet + quad)

def confusion_P_D_given_L(pred: np.ndarray, truth: np.ndarray, classes: list[int]) -> np.ndarray:
    """Column-normalized confusion: entries are P(D=i | L=j)."""
    K = len(classes)
    M = np.zeros((K, K), float)  # rows i = decision, cols j = true
    for j_idx, j in enumerate(classes):
        mask = (truth == j)
        denom = np.sum(mask)
        if denom == 0:
            continue
        for i_idx, i in enumerate(classes):
            M[i_idx, j_idx] = np.sum((pred == i) & mask) / denom
    return M

def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


# ------------------------------- PCA utils -------------------------------
def pca_fit_transform(X: np.ndarray, k: int = 3):
    """Return (scores, components). Uses SVD on centered X, no sklearn needed."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps  = Vt[:k]                 # (k, d)
    scores = Xc @ comps.T           # (n, k)
    return scores, comps

def plot_pca2(scores2: np.ndarray, y: np.ndarray, class_names: dict, title: str, out_path: str):
    plt.figure(figsize=(7,7))
    for c in np.unique(y):
        idx = (y == c)
        plt.scatter(scores2[idx,0], scores2[idx,1], s=8, alpha=0.7,
                    label=class_names.get(int(c), str(int(c))))
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title); plt.grid(True)
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close()

def plot_pca3(scores3: np.ndarray, y: np.ndarray, class_names: dict, title: str, out_path: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8,7)); ax = fig.add_subplot(111, projection='3d')
    for c in np.unique(y):
        idx = (y == c)
        ax.scatter(scores3[idx,0], scores3[idx,1], scores3[idx,2], s=8, alpha=0.7,
                   label=class_names.get(int(c), str(int(c))))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close()

def biplot_2d(scores: np.ndarray, comps: np.ndarray, feature_names: list[str], top_n: int, title: str, out_path: str):
    """Simple 2D PCA biplot: scatter + top loading arrows by length."""
    t = scores[:, :2]
    load = comps[:2, :]
    lens = np.sqrt((load**2).sum(axis=0))
    idx = np.argsort(lens)[::-1][:min(top_n, load.shape[1])]
    plt.figure(figsize=(7,7))
    plt.scatter(t[:,0], t[:,1], s=8, alpha=0.4)
    scale = 5.0 / (np.max(np.abs(load[:, idx])) + 1e-12)
    for j in idx:
        x, y = load[0, j]*scale, load[1, j]*scale
        plt.arrow(0, 0, x, y, head_width=0.08, length_includes_head=True)
        plt.text(x*1.05, y*1.05, feature_names[j], fontsize=8)
    plt.axhline(0, lw=0.5); plt.axvline(0, lw=0.5)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close()


# --------------------------- ERM / QDA / LDA ---------------------------
class GaussianERM:
    """
    ERM with Gaussian class-conditionals.
    QDA: per-class Σ_c; LDA: shared pooled Σ.
    Regularization: Σ_reg = Σ + λI,  λ = alpha * trace(Σ)/d
    """
    def __init__(self, mode: str = "QDA", alpha: float = 0.05):
        assert mode in ("QDA", "LDA")
        self.mode = mode
        self.alpha = alpha
        self.classes_ = None
        self.means_, self.covs_reg_, self.priors_ = {}, {}, {}

    def _reg(self, S: np.ndarray) -> np.ndarray:
        d = S.shape[0]
        lam = self.alpha * (np.trace(S) / d)
        return S + lam * np.eye(d)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = sorted(np.unique(y))
        n = len(y)

        # per-class means and covariances
        means = {}
        covs  = {}
        counts = {}
        for c in self.classes_:
            Xc = X[y == c]
            means[c]  = Xc.mean(axis=0)
            covs[c]   = np.cov(Xc.T, bias=False)
            counts[c] = Xc.shape[0]
            self.priors_[c] = counts[c] / n

        if self.mode == "LDA":
            d = X.shape[1]
            Sw = np.zeros((d, d))
            for c in self.classes_:
                # (unbiased) pooled within-class covariance
                Sw += covs[c] * (counts[c] - 1)
            Sw /= max(1, (n - len(self.classes_)))
            Sw_reg = self._reg(Sw)
            for c in self.classes_:
                self.means_[c]     = means[c]
                self.covs_reg_[c]  = Sw_reg
        else:
            for c in self.classes_:
                self.means_[c]     = means[c]
                self.covs_reg_[c]  = self._reg(covs[c])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = []
        for c in self.classes_:
            lp = logpdf_mvn(X, self.means_[c], self.covs_reg_[c]) + np.log(self.priors_[c])
            scores.append(lp)
        S = np.column_stack(scores)
        idx = np.argmax(S, axis=1)
        return np.array([self.classes_[i] for i in idx])


# ------------------------------- loaders -------------------------------
def load_wine_white():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    path = os.path.join(DATADIR, "winequality-white.csv")
    safe_download(url, path)
    df = pd.read_csv(path, sep=";")
    X = df.drop(columns=["quality"]).to_numpy(float)
    y = df["quality"].to_numpy(int)  # labels like 3..9
    feat_names = list(df.drop(columns=["quality"]).columns)
    return X, y, feat_names

def load_har():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    path = os.path.join(DATADIR, "UCI-HAR-Dataset.zip")
    safe_download(url, path)
    with zipfile.ZipFile(path, "r") as zf:
        base = "UCI HAR Dataset/"
        Xtr = np.loadtxt(io.BytesIO(zf.read(base + "train/X_train.txt")))
        ytr = np.loadtxt(io.BytesIO(zf.read(base + "train/y_train.txt"))).astype(int)
        Xte = np.loadtxt(io.BytesIO(zf.read(base + "test/X_test.txt")))
        yte = np.loadtxt(io.BytesIO(zf.read(base + "test/y_test.txt"))).astype(int)
    X = np.vstack([Xtr, Xte])
    y = np.hstack([ytr, yte])
    return X, y


# ------------------------------- runner -------------------------------
def run_dataset(name: str, X: np.ndarray, y: np.ndarray, feat_names: list[str] | None,
                out_dir: str, standardize: bool = False):
    print(f"\n[INFO] {name}: {X.shape[0]} samples, {X.shape[1]} features")
    os.makedirs(out_dir, exist_ok=True)

    if standardize:
        print("[INFO] Z-scoring features (mean=0, std=1)…")
        X = zscore(X)

    # Train ERM and evaluate on ALL samples (per assignment)
    print(f"[INFO] Fitting Gaussian ERM ({MODEL_MODE}) with alpha={REG_ALPHA}…")
    clf = GaussianERM(mode=MODEL_MODE, alpha=REG_ALPHA).fit(X, y)
    print("[INFO] Predicting on all samples…")
    yhat = clf.predict(X)

    # Column-normalized confusion: P(D=i | L=j)
    classes = sorted(np.unique(y))
    cm = confusion_P_D_given_L(yhat, y, classes)

    # Empirical priors from data
    priors = np.array([(y == c).mean() for c in classes], dtype=float)

    # Accuracy / error
    acc = float(np.sum(priors * np.diag(cm)))
    perror = 1.0 - acc

    # Save confusion as CSV
    conf_csv = os.path.join(out_dir, "confusion.csv")
    pd.DataFrame(cm,
                 index=[f"D={c}" for c in classes],
                 columns=[f"L={c}" for c in classes]).to_csv(conf_csv)
    print("[OK] Confusion saved ->", conf_csv)
    print(f"[OK] Empirical P(error) = {perror:.4f}")

    # PCA + plots
    print("[INFO] PCA (k=3) + plots…")
    scores3, comps3 = pca_fit_transform(X, k=3)
    names = {}
    if "har" in name.lower():
        names = {1:"WALKING", 2:"WALK_UP", 3:"WALK_DOWN", 4:"SITTING", 5:"STANDING", 6:"LAYING"}

    if PLOT_2D:
        p2 = os.path.join(out_dir, "pca2.png")
        plot_pca2(scores3[:, :2], y, names, f"{name} – PCA (2D)", p2)
        print("[OK] ->", p2)
    if PLOT_3D:
        p3 = os.path.join(out_dir, "pca3.png")
        plot_pca3(scores3[:, :3], y, names, f"{name} – PCA (3D)", p3)
        print("[OK] ->", p3)
    if PLOT_BIPLOT and feat_names is not None and "wine" in name.lower():
        pb = os.path.join(out_dir, "pca2_biplot.png")
        biplot_2d(scores3, comps3, feat_names, top_n=min(12, len(feat_names)),
                  title=f"{name} – PCA Biplot (PC1/PC2)", out_path=pb)
        print("[OK] ->", pb)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    if RUN_WINE:
        print("[INFO] Loading Wine (white)…")
        Xw, yw, feat_w = load_wine_white()
        run_dataset("Wine White", Xw, yw, feat_w, os.path.join(OUTDIR, "wine"),
                    standardize=STANDARDIZE_WINE)

    if RUN_HAR:
        print("[INFO] Loading UCI HAR…")
        Xh, yh = load_har()
        run_dataset("HAR Smartphones", Xh, yh, None, os.path.join(OUTDIR, "har"),
                    standardize=False)

    print("\n[DONE] All outputs in:", OUTDIR)


if __name__ == "__main__":
    main()
