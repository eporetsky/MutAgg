"""
Sample splitting: remove isolated outliers (k-NN in PC space), then optionally split
samples into k groups by K-means on first few PCs when silhouette score supports it.
Aims to get more homogeneous coexpression per split (e.g. roots vs shoots). Provisional.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

# Outlier: flag only isolated points (high k-th NN distance), not whole clusters
K_NEIGHBORS = 5
KNN_MAD_THRESHOLD = 4.0   # MAD multiplier; higher = fewer points called outlier
MIN_SAMPLES_TO_SPLIT = 30
MIN_CLUSTER_SIZE = 12     # MR step needs at least this many samples per input
SILHOUETTE_THRESHOLD = 0.35  # Only split if best k has silhouette above this
MAX_K = 10
N_PCS_OUTLIER = 3
N_PCS_CLUSTER = 3
SIL_MIN_AFTER_MERGE = 0.0
SIL_MIN_SMALL_CLUSTER = 0.0
DROPPED_LABEL = -1


def _open(path):
    if str(path).endswith(".gz"):
        import gzip
        return gzip.open(path, "rt")
    return open(path, "r")


def load_expression(path):
    """Load expression file: genes rows, samples columns. Returns (df, sample_names)."""
    with _open(path) as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    return df, list(df.columns)


def isolation_outliers(X_pca, n_pcs=5, k=5, mad_threshold=4.0):
    """Flag samples with unusually large distance to k-th nearest neighbor (MAD-based)."""
    n = X_pca.shape[0]
    if n <= k + 1:
        return np.zeros(n, dtype=bool)
    X = X_pca[:, : min(n_pcs, X_pca.shape[1])]
    k_use = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(X)
    dists, _ = nbrs.kneighbors(X)
    kth_dist = dists[:, k_use]
    median = np.median(kth_dist)
    mad = np.median(np.abs(kth_dist - median))
    if mad <= 0:
        return np.zeros(n, dtype=bool)
    z = (kth_dist - median) / (1.4826 * mad)
    return z > mad_threshold


def _relabel_contiguous(labels, exclude=DROPPED_LABEL):
    used = np.unique(labels)
    used = used[used != exclude]
    remap = {u: i for i, u in enumerate(sorted(used))}
    out = np.full_like(labels, exclude)
    for u in remap:
        out[labels == u] = remap[u]
    return out, len(remap)


def merge_or_drop_small_clusters(
    labels, X, min_size=12, sil_min=SIL_MIN_AFTER_MERGE, sil_min_small=SIL_MIN_SMALL_CLUSTER
):
    """Merge small clusters into nearest large cluster if silhouette allows; else drop."""
    n = len(labels)
    labels_out = np.array(labels, dtype=np.int32)
    dropped = np.zeros(n, dtype=bool)

    def sizes():
        valid = ~dropped
        if valid.sum() == 0:
            return np.array([0])
        return np.bincount(labels_out[valid], minlength=max(labels_out[valid].max() + 1, 1))

    while True:
        counts = sizes()
        small_clusters = [c for c in range(len(counts)) if 0 < counts[c] < min_size]
        if not small_clusters:
            break
        small_label = min(small_clusters, key=lambda c: counts[c])
        small_mask = (labels_out == small_label) & ~dropped
        n_small = small_mask.sum()
        if n_small == 0:
            break
        large_labels = [c for c in range(len(counts)) if counts[c] >= min_size]
        if not large_labels:
            dropped[small_mask] = True
            labels_out[small_mask] = DROPPED_LABEL
            continue
        X_valid = X[~dropped]
        lab_valid = labels_out[~dropped]
        centroids = {}
        for c in set(lab_valid) - {DROPPED_LABEL}:
            if c >= 0:
                centroids[c] = X_valid[lab_valid == c].mean(axis=0)
        small_centroid = X[small_mask].mean(axis=0)
        best_large = min(
            large_labels,
            key=lambda c: np.linalg.norm(small_centroid - centroids[c]),
        )
        lab_before, n_before = _relabel_contiguous(labels_out, exclude=DROPPED_LABEL)
        valid_before = lab_before >= 0
        sil_before = (
            silhouette_score(X[valid_before], lab_before[valid_before])
            if valid_before.sum() >= 2 and n_before >= 2
            else -1.0
        )
        labels_try = labels_out.copy()
        labels_try[small_mask] = best_large
        labels_try[dropped] = DROPPED_LABEL
        lab_remap, n_clusters = _relabel_contiguous(labels_try, exclude=DROPPED_LABEL)
        valid = lab_remap >= 0
        if valid.sum() < 2 or n_clusters < 2:
            sil_after = -1.0
            sil_small_mean = -1.0
        else:
            sil_after = silhouette_score(X[valid], lab_remap[valid])
            sil_per = silhouette_samples(X[valid], lab_remap[valid])
            valid_idx = np.where(valid)[0]
            small_in_valid = np.array([i in np.where(small_mask)[0] for i in valid_idx])
            sil_small_mean = float(sil_per[small_in_valid].mean()) if small_in_valid.any() else -1.0
        if sil_after >= sil_min and sil_small_mean >= sil_min_small:
            labels_out[small_mask] = best_large
        else:
            dropped[small_mask] = True
            labels_out[small_mask] = DROPPED_LABEL

    final_counts = np.bincount(labels_out[~dropped], minlength=labels_out.max() + 1)
    keep_labels = [
        c for c in range(len(final_counts))
        if final_counts[c] >= min_size and c != DROPPED_LABEL
    ]
    remap_final = {c: i for i, c in enumerate(sorted(keep_labels))}
    labels_final = np.full(n, DROPPED_LABEL, dtype=np.int32)
    for c in remap_final:
        labels_final[labels_out == c] = remap_final[c]
    n_dropped = int(dropped.sum())
    return labels_final, n_dropped


def run_splits(
    cpm_path,
    report_dir,
    experiment_id,
    out_dir=None,
    knn_mad=KNN_MAD_THRESHOLD,
    min_samples_split=MIN_SAMPLES_TO_SPLIT,
    min_cluster_size=MIN_CLUSTER_SIZE,
    sil_threshold=SILHOUETTE_THRESHOLD,
    sil_min_after_merge=SIL_MIN_AFTER_MERGE,
    sil_min_small=SIL_MIN_SMALL_CLUSTER,
    max_k=MAX_K,
    verbose=True,
):
    """
    Load expression, remove outliers, optionally split by K-means on PCA, write split files and reports.

    Returns (temp_dir_or_None, list of (split_id, path)). Caller should delete temp_dir when done.
    """
    df, sample_names = load_expression(cpm_path)
    X = np.log1p(df.T.values)
    n_samples = X.shape[0]

    if n_samples < 12:
        if verbose:
            print("Too few samples ({}) after load, skipping.".format(n_samples), file=sys.stderr)
        return (None, [])

    n_components = min(10, n_samples, X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    outlier_mask = isolation_outliers(
        X_pca, n_pcs=N_PCS_OUTLIER, k=K_NEIGHBORS, mad_threshold=knn_mad
    )
    n_outliers = int(outlier_mask.sum())
    clean_idx = np.where(~outlier_mask)[0]
    X_clean = X_pca[clean_idx]
    sample_names_clean = [sample_names[i] for i in clean_idx]
    n_clean = len(clean_idx)

    best_k = 1
    best_sil = -1.0
    best_labels = None
    silhouette_results = []

    # Try k=2..max_k; only accept split if silhouette > threshold and all clusters >= min size
    if n_clean >= min_samples_split:
        k_max = min(max_k + 1, n_clean // min_cluster_size + 1)
        for k in range(2, k_max):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_clean[:, :N_PCS_CLUSTER])
            counts = np.bincount(labels)
            sil_global = silhouette_score(X_clean[:, :N_PCS_CLUSTER], labels)
            sil_per = silhouette_samples(X_clean[:, :N_PCS_CLUSTER], labels)
            if verbose:
                print("  k={} sizes={} sil={:.3f}".format(k, counts.tolist(), sil_global), file=sys.stderr)
            silhouette_results.append((k, labels.copy(), sil_global, sil_per))
            if sil_global > best_sil and sil_global > sil_threshold:
                best_sil = sil_global
                best_k = k
                best_labels = labels.copy()

    if best_labels is None:
        best_labels = np.zeros(n_clean, dtype=int)
        n_dropped = 0
    else:
        best_labels, n_dropped = merge_or_drop_small_clusters(
            best_labels,
            X_clean[:, :N_PCS_CLUSTER],
            min_size=min_cluster_size,
            sil_min=sil_min_after_merge,
            sil_min_small=sil_min_small,
        )
        if verbose and n_dropped > 0:
            print("  Dropped {} samples from small cluster(s).".format(n_dropped), file=sys.stderr)

    kept = best_labels >= 0
    unique_vals = sorted(set(best_labels[kept]))
    unique_splits = [str(i + 1) for i in unique_vals]
    cluster_sizes = [(str(i + 1), int((best_labels == i).sum())) for i in unique_vals]

    os.makedirs(report_dir, exist_ok=True)

    # Reports: PCA plot, silhouettes, summary.tsv, samples.tsv
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        if n_outliers > 0:
            out_idx = np.where(outlier_mask)[0]
            ax.scatter(
                X_pca[out_idx, 0], X_pca[out_idx, 1],
                c="#888888", s=20, alpha=0.7, label="outlier", edgecolors="none",
            )
        if n_dropped > 0:
            drop_idx = clean_idx[best_labels == DROPPED_LABEL]
            ax.scatter(
                X_pca[drop_idx, 0], X_pca[drop_idx, 1],
                c="#cc6666", s=20, alpha=0.7, label="dropped (small)", edgecolors="none",
            )
        cmap = plt.get_cmap("tab10")
        for i, sid in enumerate(unique_splits):
            lab_val = int(sid) - 1
            mask = best_labels == lab_val
            idx_in_pca = clean_idx[mask]
            color = cmap((i % 10) / 9.0 if len(unique_splits) > 1 else 0)
            ax.scatter(
                X_pca[idx_in_pca, 0], X_pca[idx_in_pca, 1],
                c=[color], s=25, alpha=0.8, label=sid, edgecolors="none",
            )
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        ax.set_xlabel("PC1 ({:.1f}%)".format(var1))
        ax.set_ylabel("PC2 ({:.1f}%)".format(var2))
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(report_dir, "pca.png"), dpi=150, bbox_inches="tight")
        plt.close()

        if silhouette_results:
            n_plots = len(silhouette_results)
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
            if n_plots == 1:
                axes = [axes]
            for idx, (k, labels, sil_global, sil_per) in enumerate(silhouette_results):
                ax = axes[idx]
                y_low = 0
                for i in range(k):
                    ith_sil = np.sort(sil_per[labels == i])
                    size = ith_sil.shape[0]
                    y_high = y_low + size
                    ax.fill_betweenx(np.arange(y_low, y_high), 0, ith_sil, alpha=0.7)
                    ax.axhline(y=y_low, color="gray", linewidth=0.5)
                    y_low = y_high
                ax.set_ylabel("Sample index")
                ax.set_xlabel("Silhouette")
                ax.set_title("k={} (score={:.3f})".format(k, sil_global))
                ax.set_xlim(-0.3, 1.0)
                ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "silhouettes.png"), dpi=150, bbox_inches="tight")
            plt.close()
    except ImportError:
        pass

    summary_path = os.path.join(report_dir, "summary.tsv")
    with open(summary_path, "w") as f:
        f.write("metric\tvalue\n")
        f.write("total_samples\t{}\n".format(n_samples))
        f.write("outlier_samples\t{}\n".format(n_outliers))
        f.write("dropped_small_cluster_samples\t{}\n".format(n_dropped))
        f.write("n_clusters\t{}\n".format(len(unique_splits)))
        for sid, count in cluster_sizes:
            f.write("samples_{}\t{}\n".format(sid, count))

    orig_to_split = {}
    for j, orig_i in enumerate(clean_idx):
        if best_labels[j] == DROPPED_LABEL:
            orig_to_split[orig_i] = -2
        else:
            orig_to_split[orig_i] = str(int(best_labels[j]) + 1)
    samples_path = os.path.join(report_dir, "samples.tsv")
    with open(samples_path, "w") as f:
        f.write("sample_id\tsplit_id\n")
        for i, sid in enumerate(sample_names):
            split_id = orig_to_split.get(i, -1)
            f.write("{}\t{}\n".format(sid, split_id))

    if n_clean < 12:
        if verbose:
            print("After cleaning only {} samples (<12), skipping.".format(n_clean), file=sys.stderr)
        return (None, [])

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="mutagg_split_")
        temp_dir = out_dir
    else:
        temp_dir = None
        os.makedirs(out_dir, exist_ok=True)

    results = []
    suffix = ".tsv.gz" if str(cpm_path).endswith(".gz") else ".tsv"
    base = Path(cpm_path).stem
    if base.endswith(".tsv"):
        base = base[:-4]
    if base.endswith(".cpm"):
        base = base[:-4]

    for sid in unique_splits:
        mask = best_labels == int(sid) - 1
        sub_samples = [sample_names_clean[i] for i in np.where(mask)[0]]
        sub_df = df[sub_samples]
        out_name = "{}.{}.tsv{}".format(base, sid, ".gz" if suffix == ".tsv.gz" else "")
        out_path = os.path.join(out_dir, out_name)
        if out_path.endswith(".gz"):
            sub_df.to_csv(out_path, sep="\t", index=True, compression="gzip")
        else:
            sub_df.to_csv(out_path, sep="\t", index=True)
        results.append((sid, os.path.abspath(out_path)))

    return (temp_dir, results)
