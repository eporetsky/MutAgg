"""
Mutual Rank (MR) from expression matrix.

Computes Pearson correlation between genes, then for each pair (i,j):
rank_ij = rank of j among i's neighbors (by correlation), rank_ji = rank of i among j's.
MR(i,j) = sqrt(rank_ij * rank_ji). Lower MR = stronger coexpression.
No dependency on MutClust; self-contained implementation.
"""

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _open(path):
    """Open file, gzip if .gz."""
    path = Path(path)
    if path.suffix == ".gz" or str(path).endswith(".tsv.gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def load_expression(path, log2=False):
    """
    Load expression matrix: genes as rows, samples as columns.
    Returns (genes_index, matrix as ndarray). Optional log2(x+1) for RNA-seq counts.
    """
    with _open(path) as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    genes = np.array(df.index)
    X = df.values.astype(np.float64)
    if log2:
        X = np.log2(X + 1.0)
    return genes, X


def correlation_ranks(C):
    """
    From n x n correlation matrix C, compute for each pair (i,j) the rank of j in i's list
    (1 = highest correlation to i) and vice versa. Ranks are 1-based; self is excluded.
    Returns (rank_ij, rank_ji) as two n x n arrays where rank_ij[i,j] = rank of j in i's list.
    """
    n = C.shape[0]
    # Break ties by secondary sort on column index so order is deterministic
    # For each row i: order of columns by descending C[i,:]
    rank_ij = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = C[i, :].copy()
        row[i] = -np.inf  # exclude self from ranking
        # rank 1 = highest correlation
        order = np.argsort(-row, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        rank_ij[i, :] = ranks
    rank_ji = rank_ij.T
    return rank_ij, rank_ji


def mutual_rank_matrix(rank_ij, rank_ji):
    """MR(i,j) = geometric mean of rank_ij and rank_ji."""
    return np.sqrt(rank_ij * rank_ji)


def compute_mr(
    path,
    out_path,
    mr_threshold=100,
    log2=True,
    threads=1,
):
    """
    Compute Mutual Rank network from one expression file and write edge list.

    Parameters
    ----------
    path : str or Path
        Input expression file (.tsv or .tsv.gz), genes x samples.
    out_path : str or Path
        Output file for MR pairs (Gene1, Gene2, MR). Will be gzipped if name ends with .gz.
    mr_threshold : int
        Only output pairs with MR < this value.
    log2 : bool
        Apply log2(x+1) to expression before correlation.
    threads : int
        Unused (kept for API compatibility); correlation is single-threaded here.

    Returns
    -------
    int
        Number of edges written.
    """
    genes, X = load_expression(path, log2=log2)
    n_genes, n_samples = X.shape
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}")

    # Pearson correlation (genes x genes); NaNs from constant rows zeroed
    C = np.corrcoef(X)
    np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0, copy=False)
    np.clip(C, -1.0, 1.0, out=C)

    rank_ij, rank_ji = correlation_ranks(C)
    mr = mutual_rank_matrix(rank_ij, rank_ji)

    # Output only upper triangle and only pairs with MR below threshold
    triu_idx = np.triu_indices(n_genes, k=1)
    mr_vals = mr[triu_idx]
    r, c = triu_idx[0], triu_idx[1]
    keep = (mr_vals < mr_threshold) & np.isfinite(mr_vals)
    r, c, mr_vals = r[keep], c[keep], mr_vals[keep]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_gz = out_path.suffix == ".gz" or ".tsv.gz" in str(out_path)

    if use_gz:
        opener = gzip.open(out_path, "wt")
    else:
        opener = open(out_path, "w")

    with opener as f:
        f.write("Gene1\tGene2\tMR\n")
        for i, j, m in zip(r, c, mr_vals):
            g1, g2 = genes[i], genes[j]
            if g1 > g2:
                g1, g2 = g2, g1
            f.write(f"{g1}\t{g2}\t{m:.2f}\n")

    return len(r)
