"""
Degree-corrected aggregation of Mutual Rank coexpression across experiments.

For each gene pair we observe a count (how many experiments have that edge in the MR network).
Under a configuration-model null (degree-preserving random graph per experiment), the sum of
K Bernoulli trials with experiment-specific probabilities follows a Poisson Binomial distribution.
We test whether the observed count is higher than expected and correct for multiple testing.
Output: sparse matrix and table of significant edges (-log10(adj p-value)).
"""

import gzip
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, save_npz, triu
from scipy.stats import chi2, norm, poisson
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool
from fast_poibin import PoiBin

# Shared read-only state for worker processes (set before Pool, inherited via fork)
_degrees = None   # (K, n_genes): degree of each gene in each experiment
_two_M = None     # (K,): 2 * number of edges per experiment
_edge_rows = None
_edge_cols = None
_edge_counts = None
_lambda = 1.0     # variance inflation factor (K_eff-based or data-driven)
CHI2_1_MEDIAN = float(chi2.ppf(0.5, 1))

MR_THRESHOLD = 100
MIN_COUNT = 3
LE_CAM_THRESHOLD = 0.01   # When sum(p_k^2) < this, Poisson approx is used instead of exact PoiBin
MAX_NLOG10 = 300


def _worker(bounds):
    """Compute p-values for a chunk of edges. Null: configuration model per experiment."""
    start, end = bounds
    n = end - start
    rows = _edge_rows[start:end]
    cols = _edge_cols[start:end]
    counts = _edge_counts[start:end].astype(np.float64)
    # Configuration model: p_k(i,j) = d_i^k * d_j^k / (2*M_k)
    probs = _degrees[:, rows] * _degrees[:, cols]
    probs /= _two_M[:, np.newaxis]
    np.clip(probs, 0.0, 1.0, out=probs)
    lam = probs.sum(axis=0)
    var_null = (probs * (1.0 - probs)).sum(axis=0)
    var_null = np.maximum(var_null, 1e-12)
    _LOG10E = 1.0 / np.log(10)
    pvals = np.ones(n, dtype=np.float64)
    nlog10 = np.zeros(n, dtype=np.float64)
    if _lambda > 1.0:
        # Inflated variance: normal approximation
        var_infl = _lambda * var_null
        pos = counts > 0
        if pos.any():
            z = (counts[pos] - lam[pos]) / np.sqrt(var_infl[pos])
            pvals[pos] = np.clip(norm.sf(z), 0.0, 1.0)
            nlog10[pos] = np.clip(-norm.logsf(z) * _LOG10E, 0.0, MAX_NLOG10)
        return pvals, nlog10, 0, lam, var_null
    # Le Cam: Poisson approx when sum(p^2) small; else exact Poisson Binomial CDF
    sum_p_sq = np.einsum("ij,ij->j", probs, probs)
    poi_mask = (sum_p_sq < LE_CAM_THRESHOLD) & (counts > 0)
    if poi_mask.any():
        logsf = poisson.logsf(counts[poi_mask] - 1, lam[poi_mask])
        pvals[poi_mask] = np.exp(logsf)
        nlog10[poi_mask] = np.clip(-logsf * _LOG10E, 0.0, MAX_NLOG10)
    exact_indices = np.where((~poi_mask) & (counts > 0))[0]
    for idx in exact_indices:
        c = int(counts[idx])
        p_nz = probs[:, idx]
        p_nz = p_nz[p_nz > 0]
        if len(p_nz) == 0:
            pvals[idx] = 1.0
            nlog10[idx] = 0.0
        elif c > len(p_nz):
            pvals[idx] = 0.0
            nlog10[idx] = MAX_NLOG10
        else:
            cdf_val = PoiBin(p_nz.tolist()).cdf[c - 1]
            pvals[idx] = max(1.0 - cdf_val, 0.0)
            nlog10[idx] = min(-np.log1p(-cdf_val) * _LOG10E, MAX_NLOG10) if cdf_val < 1.0 else MAX_NLOG10
    return pvals, nlog10, len(exact_indices), lam, var_null


def _open_mr(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def get_gene_list_from_mr_files(mr_paths):
    """Collect sorted union of genes from all MR files."""
    genes = set()
    for p in mr_paths:
        with _open_mr(p) as f:
            df = pd.read_csv(f, sep="\t")
        for col in ("Gene1", "Gene2"):
            if col in df.columns:
                genes.update(df[col].astype(str))
    return sorted(genes)


def load_mr_network(path, idx_map, mr_thresh=MR_THRESHOLD):
    """Load one MR file into symmetric sparse adjacency (same gene order as idx_map)."""
    with _open_mr(path) as f:
        df = pd.read_csv(f, sep="\t")
    if "MR" not in df.columns:
        raise ValueError("MR file must have columns Gene1, Gene2, MR")
    df = df[df["MR"] < mr_thresh]
    g1, g2 = df["Gene1"].values.astype(str), df["Gene2"].values.astype(str)
    r = np.array([idx_map.get(min(a, b), -1) for a, b in zip(g1, g2)])
    c = np.array([idx_map.get(max(a, b), -1) for a, b in zip(g1, g2)])
    keep = (r >= 0) & (c >= 0)
    r, c = r[keep], c[keep]
    n = len(idx_map)
    mat = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    return (mat + mat.T).tocsr()


def run_aggregation(
    mr_dir,
    out_dir,
    species_name="default",
    gene_list=None,
    mr_threshold=MR_THRESHOLD,
    min_count=MIN_COUNT,
    use_var_inflation=False,
    mt_method="bonferroni",
    hist=False,
    n_workers=None,
    log_file=None,
):
    """
    Aggregate MR networks for one species: load all .mr.tsv.gz in mr_dir,
    run degree-corrected Poisson Binomial test, write .aggregated.npz and .aggregated.tsv to out_dir.

    gene_list: if None, inferred from union of genes in MR files.
    mt_method: 'bonferroni' or 'fdr'.
    """
    global _degrees, _two_M, _edge_rows, _edge_cols, _edge_counts, _lambda

    mr_dir = Path(mr_dir)
    out_dir = Path(out_dir)
    mr_files = sorted(mr_dir.glob("*.mr.tsv.gz")) + sorted(mr_dir.glob("*.mr.tsv"))
    mr_files = [str(p) for p in mr_files]
    K = len(mr_files)
    if K == 0:
        raise FileNotFoundError("No MR files in {}".format(mr_dir))

    if gene_list is None:
        gene_list = get_gene_list_from_mr_files(mr_files)
    idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)

    def log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    log("\n" + "=" * 60)
    log("{}: {} genes x {} experiments  (MR < {}, count >= {})".format(
        species_name, n_genes, K, mr_threshold, min_count))
    log("=" * 60)

    # Accumulate edge counts and per-experiment degrees (for configuration-model null)
    count_mat = csr_matrix((n_genes, n_genes), dtype=np.int16)
    degrees = np.zeros((K, n_genes), dtype=np.float64)
    two_M = np.zeros(K, dtype=np.float64)

    for k, mrf in enumerate(mr_files):
        t0 = time.time()
        mat = load_mr_network(mrf, idx, mr_thresh=mr_threshold)
        deg = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
        degrees[k] = deg
        two_M[k] = deg.sum()
        count_mat = count_mat + mat
        log("  [{:>3}/{}] {:40s} {:>10,} edges  ({:.1f}s)".format(
            k + 1, K, os.path.basename(mrf), int(two_M[k]) // 2, time.time() - t0))

    two_M[two_M == 0] = 1.0

    # Upper triangle only; keep edges seen in >= min_count experiments
    ut = triu(count_mat, k=1).tocoo()
    keep = ut.data >= min_count
    _edge_rows = ut.row[keep].astype(np.int32)
    _edge_cols = ut.col[keep].astype(np.int32)
    _edge_counts = ut.data[keep].astype(np.int32)
    n_edges = len(_edge_rows)
    n_before = len(ut.data)
    log("\n  {:,} unique edges total, {:,} with count >= {}".format(n_before, n_edges, min_count))

    # Variance inflation: effective experiment count K_eff from correlation of degree vectors
    _lambda = 1.0
    if not use_var_inflation:
        C = np.corrcoef(degrees)
        C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(C, -1.0, 1.0, out=C)
        evals = np.linalg.eigvalsh(C)
        evals = np.maximum(evals, 0.0)
        sum_sq = (evals ** 2).sum()
        K_eff = (K ** 2) / sum_sq if sum_sq > 0 else K
        _lambda = K / K_eff  # inflate null variance when experiments are correlated
        log("  Variance inflation λ = {:.4f} (effective experiments K_eff = {:.1f})".format(_lambda, K_eff))
    _degrees = degrees
    _two_M = two_M

    n_workers = n_workers or min(os.cpu_count() or 1, 70)
    cs = min(50_000, max(10_000, n_edges // (n_workers * 16)))
    ranges = [(s, min(s + cs, n_edges)) for s in range(0, n_edges, cs)]
    log("  {} workers, {} chunks (~{:,} edges/chunk)".format(n_workers, len(ranges), cs))

    t0 = time.time()
    total_exact = 0
    lam_parts, var_parts = [], []
    pval_parts, nlog10_parts = [], []

    with Pool(n_workers) as pool:
        for pv, nl, n_ex, lam_chunk, var_chunk in pool.imap(_worker, ranges):
            pval_parts.append(pv)
            nlog10_parts.append(nl)
            lam_parts.append(lam_chunk)
            var_parts.append(var_chunk)
            total_exact += n_ex

    pvals = np.concatenate(pval_parts)
    nlog10_raw = np.concatenate(nlog10_parts)
    lam_all = np.concatenate(lam_parts)
    var_null_all = np.concatenate(var_parts)
    dt = time.time() - t0
    log("  {:.1f}s total  ({:,.0f} edges/s)".format(dt, n_edges / max(dt, 0.01)))
    log("  {:,} edges needed exact PoiBin".format(total_exact))

    if use_var_inflation:
        eps = 1e-12
        use = var_null_all > eps
        stat = np.full(n_edges, np.nan, dtype=np.float64)
        stat[use] = (_edge_counts[use].astype(np.float64) - lam_all[use]) ** 2 / var_null_all[use]
        median_stat = np.nanmedian(stat[use])
        _lambda = median_stat / CHI2_1_MEDIAN
        log("  Variance inflation λ = {:.4f} (from data)".format(_lambda))
        if _lambda > 1.0:
            z = (_edge_counts.astype(np.float64) - lam_all) / np.sqrt(_lambda * var_null_all)
            pvals = np.clip(norm.sf(z), 0.0, 1.0)
            nlog10_raw = np.clip(-norm.logsf(z) * (1.0 / np.log(10)), 0.0, MAX_NLOG10)

    mt_method_internal = "fdr_bh" if mt_method == "fdr" else "bonferroni"
    mt_label = "FDR (BH)" if mt_method_internal == "fdr_bh" else "Bonferroni"
    log("  Applying {} correction …".format(mt_label))
    _, adj_pvals, _, _ = multipletests(pvals, alpha=0.05, method=mt_method_internal)

    if mt_method_internal == "bonferroni":
        nlog10_adj_all = np.maximum(nlog10_raw - np.log10(n_edges), 0.0)
    else:
        order = np.argsort(-nlog10_raw, kind="stable")
        nlog10_sorted = nlog10_raw[order]
        log10_m = np.log10(n_edges)
        ranks = np.arange(1, n_edges + 1, dtype=np.float64)
        nlog10_bh = nlog10_sorted - log10_m + np.log10(ranks)
        nlog10_bh = np.maximum.accumulate(nlog10_bh[::-1])[::-1].copy()
        np.maximum(nlog10_bh, 0.0, out=nlog10_bh)
        nlog10_adj_all = np.empty(n_edges, dtype=np.float64)
        nlog10_adj_all[order] = nlog10_bh

    sig_mask = adj_pvals < 0.05
    sig_rows = _edge_rows[sig_mask]
    sig_cols = _edge_cols[sig_mask]
    sig_adj = adj_pvals[sig_mask]
    sig_raw = pvals[sig_mask]
    sig_counts = _edge_counts[sig_mask]
    sig_nlog10_raw = nlog10_raw[sig_mask]
    sig_nlog10_adj = nlog10_adj_all[sig_mask]
    n_sig = int(sig_mask.sum())

    log("  {:,} edges pass adj p < 0.05 (of {:,} tested)".format(n_sig, n_edges))

    nlp = np.where(np.isfinite(sig_nlog10_adj), sig_nlog10_adj, 300.0).astype(np.float32)
    all_r = np.concatenate([sig_rows, sig_cols])
    all_c = np.concatenate([sig_cols, sig_rows])
    all_d = np.concatenate([nlp, nlp])
    result = coo_matrix((all_d, (all_r, all_c)), shape=(n_genes, n_genes)).tocsr()

    os.makedirs(out_dir, exist_ok=True)
    out_npz = Path(out_dir) / (str(species_name).replace(".npz", "").replace(".tsv", "") + ".aggregated.npz")
    save_npz(out_npz, result)
    log("  -> {}".format(out_npz))

    gene_arr = np.array(gene_list)
    nlog10_pval_out = np.where(np.isfinite(sig_nlog10_raw), sig_nlog10_raw, MAX_NLOG10)
    nlog10_adj_out = np.where(np.isfinite(sig_nlog10_adj), sig_nlog10_adj, MAX_NLOG10)
    tsv_df = pd.DataFrame({
        "Gene1": gene_arr[sig_rows],
        "Gene2": gene_arr[sig_cols],
        "count": sig_counts,
        "pval": sig_raw,
        "adj_pval": sig_adj,
        "nlog10_pval": nlog10_pval_out,
        "nlog10_adj": nlog10_adj_out,
    })
    tsv_df.sort_values("adj_pval", inplace=True)
    tsv_path = Path(out_dir) / (str(species_name).replace(".tsv", "").replace(".npz", "") + ".aggregated.tsv")
    tsv_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.4e")
    log("  -> {} ({:,} rows)".format(tsv_path, len(tsv_df)))

    if hist:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            count_bins = max(2, min(50, int(_edge_counts.max()) - int(_edge_counts.min()) + 1))
            ax1.hist(_edge_counts, bins=count_bins, color="#1f77b4", alpha=0.7, edgecolor="black")
            ax1.set_xlabel("Recurrence count (experiments per edge)")
            ax1.set_ylabel("Number of edges")
            ax1.set_title("Count distribution")
            ax1.axvline(_edge_counts.mean(), color="red", linestyle="--", label="mean = {:.2f}".format(_edge_counts.mean()))
            ax1.legend()
            eps = 1e-12
            use = var_null_all > eps
            stat = (_edge_counts[use].astype(np.float64) - lam_all[use]) ** 2 / var_null_all[use]
            stat_cap = np.minimum(stat, 15.0)
            ax2.hist(stat_cap, bins=80, density=True, color="#1f77b4", alpha=0.7, label="Observed (capped at 15)")
            x = np.linspace(0.001, 15, 300)
            ax2.plot(x, chi2.pdf(x, 1), "r-", lw=2, label="χ²(1) null")
            ax2.axvline(np.median(stat), color="green", linestyle="--", label="median = {:.3f}".format(np.median(stat)))
            ax2.set_xlabel("(count − λ)² / V")
            ax2.set_ylabel("Density")
            ax2.set_title("Residual squared (λ = {:.3f})".format(_lambda))
            ax2.legend(fontsize=8)
            fig.tight_layout()
            hist_path = Path(out_dir) / (str(species_name).replace(".png", "").replace(".tsv", "").replace(".npz", "") + ".aggregated.hist.png")
            fig.savefig(hist_path, dpi=150)
            plt.close(fig)
            log("  -> {}".format(hist_path))
        except ImportError:
            log("  (--hist: install matplotlib to generate histogram)")
