# MutAgg

MutAgg runs a **Mutual Rank (MR)** coexpression pipeline on gene expression matrices and **aggregates** results across many experiments with a **degree-corrected** statistical test. It is built for meta-analysis of coexpression: you have multiple RNA-seq studies (e.g. different accessions, tissues, or conditions), each yielding an expression matrix (genes × samples). MutAgg computes MR networks per experiment (or per sample-split when using the optional K-means splitting), then asks which gene pairs are coexpressed **more often than expected by chance** across experiments, while correcting for the fact that highly connected “hub” genes will share many partners by chance alone.

---

## Disclaimer

**This approach has not been benchmarked.** Use at your own discretion. The method is provided as-is for research and exploratory analysis.

---

## What MutAgg does

1. **Mutual Rank (MR)**  
   From each expression matrix (genes × samples), compute Pearson correlation between genes, then for each pair *(i, j)* the rank of *j* among *i*’s neighbors and the rank of *i* among *j*’s neighbors. The **Mutual Rank** is the geometric mean of these two ranks; lower MR means stronger coexpression. Only pairs with MR below a threshold are kept.  
   *Useful because* it focuses on reciprocal, strong coexpression and keeps the network sparse.

2. **Optional sample splitting (K-means)**  
   Before MR, each experiment can be split into more homogeneous sample groups: remove isolated outliers (k-NN in PCA space), then run K-means on the first few PCs and choose *k* by silhouette score. Each resulting group is treated as a separate “experiment” for MR and aggregation. **This step is provisional** (see below).  
   *Useful because* it can separate distinct conditions (e.g. roots vs leaves) so coexpression is estimated within more homogeneous samples.

3. **Degree correction and Poisson Binomial test**  
   Across experiments we observe, for each gene pair, a **count**: how many experiments have that edge in the MR network. Hub genes have many edges per experiment and will get high counts by chance. MutAgg assigns each edge a **null probability per experiment** from the node degrees: *pₖ(i,j) = dᵢᵏ dⱼᵏ / (2 Mₖ)* (so high-degree genes have higher *pₖ*). The total count is then tested against a **Poisson Binomial** (sum of K Bernoulli trials with these different *pₖ*).  
   *Useful because* the same count can be non-significant for a hub pair and significant for a specialist pair, so rankings reflect biological signal rather than connectivity.

4. **Variance inflation**  
   When experiments are correlated (e.g. similar tissues or batches), *p*-values can be over-confident. MutAgg inflates the null variance using an **effective experiment count** *K_eff* derived from the correlation of degree vectors across experiments: λ = *K* / *K_eff*. Optionally, a data-driven λ can be used instead.  
   *Useful because* it reduces false positives when you have many similar experiments.

After multiple-testing correction (Bonferroni or FDR), only edges with adjusted *p* < 0.05 are kept. Output: per “species” (or group), an **aggregated** sparse matrix and TSV of significant edges (count, *p*-value, adjusted *p*-value, -log₁₀ values).

---

## Statistical details

- **Null probability:** For experiment *k*, *pₖ(i,j) = dᵢᵏ dⱼᵏ / (2 Mₖ)* (degree-corrected: higher degree → higher chance of edge by chance).
- **Test:** Sum of K independent Bernoulli(*pₖ*) → Poisson Binomial. One-sided *P(X ≥ observed count)*.
- **Computation:** When ∑*pₖ²* is small (Le Cam), Poisson(∑*pₖ*) approximates the Poisson Binomial; otherwise the exact CDF is used (fast_poibin).
- **Variance inflation:** *K_eff = K² / ∑λᵢ²* from the experiment–experiment correlation matrix of degree vectors; λ = *K* / *K_eff* inflates the null variance. Optional: data-driven λ from (count − λ)² / V vs χ²(1).
- **Multiple testing:** Bonferroni (default) or Benjamini–Hochberg FDR. Only edges with adjusted *p* < 0.05 are retained.

---

## K-means sample splitting (provisional)

Before MR, MutAgg can **split each expression file** into sub-experiments by clustering samples:

1. **Outlier removal:** PCA on log-expression, then flag **isolated** samples (high k-NN distance in PC space, MAD-based). Whole clusters (e.g. a distinct tissue) are kept; only singleton/few technical outliers are removed.
2. **K-means on PCs:** Try *k* = 2 … max_k. For each *k*, run K-means on the first few PCs and compute silhouette score. Split only if the best *k* has silhouette above a threshold (default 0.35) and every cluster has at least 12 samples.
3. **Small clusters:** Clusters with &lt; 12 samples are either merged into the nearest large cluster (if silhouette criteria allow) or dropped.

Each resulting group is written as a temporary expression matrix; MR is run on each; then the temp files are removed. Aggregation then sees more, finer “experiments,” which can improve power and homogeneity.

**Status:** This splitting strategy appears to work in practice (e.g. separating roots vs leaves, or conditions), but it **has not been formally benchmarked**. Treat it as exploratory; use `--no-split` to run MR on each file as-is if you prefer.

---

## Installation

```bash
pip install mutagg
```

Optional (for report plots and aggregation histograms):

```bash
pip install mutagg[plots]
```

**Requirements:** Python 3.9+, numpy, pandas, scipy, scikit-learn, statsmodels, fast-poibin.

---

## Usage

**Full pipeline** (expression directory → optional split → MR → aggregation):

```bash
mutagg run --input-dir /path/to/expression/files --output-dir /path/to/output
```

- **Input:** A directory containing `.tsv` or `.tsv.gz` expression files (genes × samples). Row index = gene ID, columns = sample IDs.
- **Output:** Under `--output-dir`:
  - `MR/` — Mutual Rank edge lists (Gene1, Gene2, MR) per experiment (and per split if not `--no-split`).
  - `reports/` — Per-experiment reports: PCA plot, silhouettes, `summary.tsv`, `samples.tsv`.
  - `aggregated/` — `{species}.aggregated.npz`, `{species}.aggregated.tsv`, and `aggregation.log`.

### Options

| Option | Description |
|--------|-------------|
| `--input-dir`, `-i` | Directory of expression files (required). |
| `--output-dir`, `-o` | Output directory (required). |
| `--recursive`, `-r` | Treat subdirectories as species (one species per subdir). |
| `--no-split` | Skip sample splitting; run MR on each file as-is. |
| `--mr-threshold`, `-m` | Only output MR pairs with MR < this (default 100). |
| `--min-count` | Min experiments per edge for aggregation (default 3). |
| `--mt-method` | Multiple testing: `bonferroni` or `fdr` (default bonferroni). |
| `--var` | Use data-driven variance inflation instead of K_eff. |
| `--hist` | Write aggregation histogram PNG per species. |
| `--no-log2` | Do not apply log2(x+1) to expression before MR. |

The default parameters (MR threshold 100, min-count 3, Bonferroni, K_eff-based variance inflation, log2 transform) are recommended for typical use; change them only if you have a specific reason.

### Examples

```bash
# All files in one group ("default"), with sample splitting and aggregation
mutagg run -i ./data/expression -o ./results

# Subdirs = species, no splitting, FDR correction
mutagg run -i ./data -o ./out --recursive --no-split --mt-method fdr
```

---

## Pipeline summary

1. **Discover expression files** in `--input-dir` (optionally per-species with `--recursive`).
2. **Optional split:** For each file, remove isolated outliers (k-NN), optionally split samples by K-means on PCA (silhouette-based); write split matrices to a temp dir and reports to `reports/{species}/{experiment}/`.
3. **MR:** Compute Mutual Rank from each (possibly split) expression matrix; write `MR/{species}/{experiment}.mr.tsv.gz` (or `{experiment}.{split_id}.mr.tsv.gz`).
4. **Aggregation:** Load all MR networks per species; degree-corrected Poisson Binomial test; output `aggregated/{species}.aggregated.npz` (sparse -log10(adj p)) and `aggregated/{species}.aggregated.tsv` (significant edges table).

---

## License

MIT.
