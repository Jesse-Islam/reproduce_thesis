import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text
import anndata as ad


import pandas as pd

def filter_primary_tumors(expr_df, gene_id_col='gene_id'):
    """
    Return only the TCGA primary-tumor (01) columns, plus the gene_id column.
    """
    # all columns except gene_id
    sample_cols = [c for c in expr_df.columns if c != gene_id_col]
    # select only those whose 4th hyphen field == "01"
    primary = [bc for bc in sample_cols if bc.split('-')[3] == '01']
    # make sure gene_id is first
    keep = [gene_id_col] + primary
    return expr_df.loc[:, keep]

def filter_smallest_sample(expr_df, gene_id_col='gene_id'):
    """
    Return only one column per unique sample (grouped by the part before the sample code),
    choosing the smallest code in that sample‐type field.
    """
    # all columns except gene_id
    sample_cols = [c for c in expr_df.columns if c != gene_id_col]

    # map from prefix -> chosen barcode
    prefix_map = {}
    for bc in sample_cols:
        # split off the last hyphen: everything before is the "prefix",
        # everything after is the sample‐type code
        prefix, code_str = bc.rsplit('-', 1)

        # try numeric comparison, fallback to lexicographic
        try:
            code_val = int(code_str)
            prev_val = int(prefix_map[prefix].rsplit('-', 1)[1]) if prefix in prefix_map else None
        except ValueError:
            code_val = code_str
            prev_val = prefix_map[prefix].rsplit('-', 1)[1] if prefix in prefix_map else None

        # keep this barcode if it's the first one for this prefix,
        # or if its code is smaller than the one we stored
        if prefix not in prefix_map or code_val < prev_val:
            prefix_map[prefix] = bc

    # build the final column list, with gene_id first
    keep_cols = [gene_id_col] + list(prefix_map.values())
    return expr_df.loc[:, keep_cols]


def align_on_barcodes(
    rnaseq: pd.DataFrame,
    clinical: pd.DataFrame,
    barcode_col: str = 'bcr_patient_barcode',
    gene_id_col: str = 'gene_id'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align an RNA-seq (genes x samples) and a clinical table by the first 12 characters
    of their sample barcodes. Reports sample counts & mismatches, and returns the
    subsetted DataFrames containing only the common samples.

    Parameters
    ----------
    rnaseq : pd.DataFrame
        Raw RNA-seq DataFrame where the first column is gene_id and all other
        columns are sample IDs (containing >=12-char barcodes).
    clinical : pd.DataFrame
        Clinical metadata, with one column equal to `barcode_col`.
    barcode_col : str, optional
        Name of the clinical column containing full patient barcodes.
    gene_id_col : str, optional
        Name of the first column in `rnaseq` holding gene IDs.

    Returns
    -------
    (rnaseq_sub, clinical_sub) : tuple of pd.DataFrames
        Subsets of the inputs restricted to samples present in both tables.
    """
    # 1) Extract barcodes from the rnaseq columns
    sample_cols = rnaseq.columns.drop(gene_id_col)
    rnaseq_barcodes = (
        pd.Series(sample_cols)
          .str[:12]
          .str.upper()
          .unique()
    )
    set_r = set(rnaseq_barcodes)

    # 2) Normalize clinical barcodes
    clinical_barcodes = clinical[barcode_col].str.upper().unique()
    set_c = set(clinical_barcodes)

    # 3) Compute intersection and differences
    common = set_r & set_c
    missing_in_clinical = set_r - set_c
    missing_in_rnaseq  = set_c - set_r

    # 4) Report
    print(f"• RNA-seq samples:   {len(set_r)}")
    print(f"• Clinical samples: {len(set_c)}")
    print(f"• Overlap:          {len(common)}")
    if missing_in_clinical:
        print(f"  – {len(missing_in_clinical)} barcodes in RNA-seq not in clinical, e.g.:",
              list(missing_in_clinical)[:5], "…")
    if missing_in_rnaseq:
        print(f"  – {len(missing_in_rnaseq)} barcodes in clinical not in RNA-seq, e.g.:",
              list(missing_in_rnaseq)[:5], "…")

    # 5) Subset rnaseq: keep gene_id col + only matching sample cols
    keep_cols = [gene_id_col] + [
        col for col in sample_cols
        if col[:12].upper() in common
    ]
    rnaseq_sub = rnaseq.loc[:, keep_cols].copy()

    # 6) Subset clinical to the common barcodes
    clinical_sub = clinical.loc[
        clinical[barcode_col].str.upper().isin(common)
    ].copy()

    return rnaseq_sub, clinical_sub

def build_anndata(rnaseq: pd.DataFrame, clinical: pd.DataFrame) -> ad.AnnData:
    """
    Process bulk RNA-seq and clinical data into a normalized AnnData object, including sample type codes.

    Parameters:
        rnaseq (pd.DataFrame): Raw EB++-normalized RNA-seq data (genes x samples).
        clinical (pd.DataFrame): Clinical metadata with 'bcr_patient_barcode'.

    Returns:
        ad.AnnData: Aligned and log-normalized AnnData object with sample_code in obs.
    """

    print("Loaded RNA-seq data:", rnaseq.shape)
    print("Loaded clinical data:", clinical.shape)

    # === Step 1: Clean gene IDs ===
    rnaseq = rnaseq.copy()
    rnaseq['gene_symbol'] = rnaseq['gene_id'].copy()
    rnaseq = rnaseq.set_index('gene_symbol')

    # === Step 2: Transpose to samples x genes ===
    expr = rnaseq.drop(columns=['gene_id']).T
    expr.index.name = 'sample_id'
    print("Transposed expression matrix:", expr.shape)

    # === Step 3: Extract patient barcodes and sample codes ===
    expr['bcr_patient_barcode'] = expr.index.str.slice(0, 12).str.upper()
    expr['sample_code'] = expr.index.str.slice(13, 15)  # two-digit code at positions 4th field

    # === Step 4: Prepare clinical data ===
    clinical = clinical.copy()
    clinical['bcr_patient_barcode'] = clinical['bcr_patient_barcode'].str.upper()

    # === Step 5: Merge expression with clinical ===
    merged = expr.merge(clinical, on='bcr_patient_barcode', how='inner')
    print("Merged expression and clinical data:", merged.shape)
    print(f"Matched patients: {merged['bcr_patient_barcode'].nunique()} / {clinical['bcr_patient_barcode'].nunique()}")

    # === Step 6: Extract expression matrix and metadata ===
    X = merged.drop(columns=list(clinical.columns) + ['bcr_patient_barcode', 'sample_code'])

    # === Step 7: Drop genes (columns) with any NaNs ===
    print("Removing genes with NaNs...")
    X_clean = X.dropna(axis=1)
    print(f"Remaining genes after NaN removal: {X_clean.shape[1]}")

    #if X_clean.columns.duplicated().any():
    #    X_clean = X_clean.groupby(X_clean.columns, axis=1).mean()
    #    print(f"Collapsed duplicates by averaging → now {X_clean.shape[1]} unique genes")

    # Already log-transformed, so no re-log
    X_final = X_clean

    print(X_final.shape)
    # === Step 9: Create obs and var DataFrames ===
    obs = merged[list(clinical.columns) + ['sample_code']].copy()
    obs.index = merged.index

    var = pd.DataFrame(index=X_final.columns)
    var.index.name = 'gene_symbol'

    # === Step 10: Create AnnData ===
    adata = ad.AnnData(X=X_final.values, obs=obs, var=var)
    adata.obs_names = obs.index
    adata.var_names = var.index

    print("AnnData created!")
    print(f"Shape: {adata.shape}")
    print(f"Samples: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")
    assert all(adata.obs_names == obs.index), "obs names do not match X rows!"
    assert all(adata.var_names == X_final.columns), "var names do not match X columns!"
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # === Step 11: Sanity checks ===

    assert not pd.isnull(adata.X).any(), "Missing values in expression matrix!"

    print("Missing values in obs:", adata.obs.isnull().sum().sum())
    print("Missing values in var:", adata.var.isnull().sum().sum())

    print("\nobs preview:\n", adata.obs.head())
    print("\nvar preview:\n", adata.var.head())
    print("\nExpression matrix shape:", adata.X.shape)

    # === Step 12: Filter patients with missing PFI metrics ===
    adata = adata[~adata.obs['PFI'].isna() & ~adata.obs['PFI.time'].isna()].copy()

    return adata


def categorize_ajcc_risk(stages: pd.Series) -> pd.Series:
    """
    Categorize AJCC pathological tumor stages into 'High Risk' or 'Low Risk'.

    Parameters:
        stages (pd.Series): A Series of AJCC stage labels (e.g., 'Stage I', 'Stage IIIC').

    Returns:
        pd.Series: A Series of risk categories ('Low Risk' or 'High Risk').
    """
    # Very low-risk stages (truly early, in situ or localized)
    low_risk = {
        "Stage 0", "Stage I", "Stage IA", "Stage IB", "IS"
    }

    # Everything else is considered high risk (including ambiguous/intermediate stages)
    high_risk = {"Stage IIA", "Stage IIB", "Stage IIC", "I/II NOS",
        "Stage II","Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC",
        "Stage IV", "Stage IVA", "Stage IVB", "Stage IVC"
    }

    def classify(stage):
        if stage in low_risk:
            return "_Low Risk"
        elif stage in high_risk:
            return "_High Risk"
        else:
            return "Unknown"  # Catch-all for anything not explicitly listed

    return stages.apply(classify)


def group_by_both_obs(adata, col1: str, col2: str) -> pd.DataFrame:
    """
    Group adata.obs by two columns and return a contingency table.
    
    Parameters:
        adata (AnnData): The AnnData object.
        col1 (str): The first column to group by.
        col2 (str): The second column to group by.
    
    Returns:
        pd.DataFrame: A contingency table with counts for each combination of col1 and col2.
    """
    # Check that the columns exist in adata.obs
    for col in [col1, col2]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")
    
    # Create a contingency table using pd.crosstab
    contingency = pd.crosstab(adata.obs[col1], adata.obs[col2], dropna=False)

    
    return contingency


def extract_rank_genes_groups(adata, key='hi', n_genes=None):
    """
    Extracts the ranked genes from the `rank_genes_groups` results stored in adata.uns 
    and returns them as a pandas DataFrame.
    
    Parameters:
    - adata: AnnData object containing the single-cell data.
    - key: The key under which the rank_genes_groups results are stored (default is 'hi').
    - n_genes: Number of top genes to extract for each group. If None, all genes are extracted.
    
    Returns:
    - A pandas DataFrame with the ranked genes, log fold changes, p-values, adjusted p-values, 
      and the group (cluster) information.
    """
    
    # Extract the results from the adata.uns dictionary
    rank_genes = adata.uns[key]
    
    # Get the groups (clusters or cell types)
    groups = rank_genes['names'].dtype.names
    
    result_dfs = {}
    
    # Loop through each group and extract relevant statistics
    for group in groups:
        # Extract the top `n_genes` if specified, otherwise take all genes
        if n_genes is not None:
            df = pd.DataFrame({
                'gene': rank_genes['names'][group][:n_genes],
                'scores_'+ key: rank_genes['scores'][group][:n_genes],
                'logfoldchange': rank_genes['logfoldchanges'][group][:n_genes],
                'pval': rank_genes['pvals'][group][:n_genes],
                'pval_adj': rank_genes['pvals_adj'][group][:n_genes],
            })
        else:
            df = pd.DataFrame({
                'gene': rank_genes['names'][group],
                'scores_'+ key: rank_genes['scores'][group],
                'logfoldchange': rank_genes['logfoldchanges'][group],
                'pval': rank_genes['pvals'][group],
                'pval_adj': rank_genes['pvals_adj'][group],
            })
        
        result_dfs[group] = df

    # Concatenate the results from all groups into a single DataFrame
    final_df = pd.concat([df.assign(group=group) for group, df in result_dfs.items()], ignore_index=True)
    
    return final_df

def merge_adata(adata1,adata2):
    adata1=adata1.copy()
    adata2=adata2.copy()
    common_genes = adata1.var.index.intersection(adata2.var.index)
    adata1 = adata1[:, common_genes]
    adata2 = adata2[:, common_genes]
    adata_combined = sc.concat([adata1, adata2], join='inner', label='batch')
    return adata_combined

def generate_wheel_correlation_matrix(n):
    cor_amount=0.7
    """
    Generates a correlation matrix with a wheel structure.
    
    Parameters:
        n (int): Total number of features in the matrix. 
                 Must be >= 4 to form a wheel structure.
    
    Returns:
        np.ndarray: A correlation matrix with a wheel-like structure.
    """
    if n < 4:
        raise ValueError("Number of features (n) must be >= 4 to form a wheel structure.")
    
    # Initialize the correlation matrix with zeros
    corr_matrix = np.zeros((n, n))
    
    # The central node (0) is fully connected to all others with correlation = 1
    corr_matrix[0, 1:] = cor_amount
    corr_matrix[1:, 0] = cor_amount
    
    # Create the cycle for the peripheral nodes
    for i in range(1, n - 1):
        corr_matrix[i, i + 1] = cor_amount # Correlation between neighbors in the cycle
        corr_matrix[i + 1, i] = cor_amount
        
    # Connect the last node in the cycle back to the first peripheral node
    corr_matrix[1, n - 1] = cor_amount
    corr_matrix[n - 1, 1] = cor_amount
    
    # Diagonal elements are 1 (self-correlation)
    np.fill_diagonal(corr_matrix, 1)
    
    return corr_matrix



def inject_network(data, para, mat_key, top_genes,biggest_change_gene, status_col='state'):
    """
    Process start and goal means based on the biggest change gene.

    Parameters:
    - data: DataFrame containing the `newCovariate` column used for filtering.
    - para: Dictionary containing the `mat` (or an equivalent key-value).
    - mat_key: Key in the `para` dictionary for the matrix to use.
    - boost_set: Number of top genes to consider based on `absolute_scores`. Default is 10.
    - status_col: The column in `data['newCovariate']` indicating 'start' or 'goal'. Default is 'status'.

    Returns:
    - start: DataFrame with modified start means.
    - goal: DataFrame with modified goal means.
    - everything gets the top differentially expressed gene's parameters, in the defined list top_genes.
    """
    mat = para[mat_key].copy()
    # Filter for start and goal means
    start = mat[data['newCovariate'][status_col] == "start"].copy()
    temp = mat[data['newCovariate'][status_col] == "start"].copy()
    goal = mat[data['newCovariate'][status_col] == "goal"].copy()
    # Update means based on the biggest change gene
    for gene in top_genes:
        start.loc[:, gene] = start.loc[:, biggest_change_gene].mean().copy()
        temp.loc[:, gene] = goal.loc[:, biggest_change_gene].mean().copy()
        
    # Update goal
   
    
    goal = temp.copy()
    
    return start, goal



def is_symmetric(M, tol=1e-8):
    return np.allclose(M, M.T, atol=tol)
from scipy.linalg import eigh

def is_psd(M, tol=0):
    # eigh returns ascending eigenvalues for symmetric M
    vals = eigh(M, eigvals_only=True)
    return (vals >= -tol).all()

def classify_matrix(M, tol=1e-8):
    if not is_symmetric(M, tol):
        return "neither (not symmetric)"
    if not is_psd(M, tol):
        return "indefinite (not PSD)"
    diag = np.diag(M)
    if np.allclose(diag, 1, atol=tol):
        return "correlation matrix"
    else:
        return "covariance matrix"


def plot_sorted_selected_labels(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    highlight_labels: list = None,
    ci: float = 0.80,
    figsize=(12,6),
):
    # ---- 0) sanity checks ----
    if group_col not in df.columns:
        raise KeyError(f"group_col {group_col!r} not found in DataFrame columns")

    # ---- 1) aggregate ----
    grouped = df.groupby(group_col)[value_col]
    means  = grouped.mean()
    n      = grouped.count()
    sems   = grouped.sem(ddof=1)

    # t-based half-width
    alpha = 1 - ci
    t_mult = n.apply(lambda ni: stats.t.ppf(1 - alpha/2, ni-1) if ni > 1 else 0.0)
    ci_err = t_mult * sems

    agg = pd.DataFrame({'mean': means, 'ci': ci_err})
    agg = agg.sort_values('mean', ascending=False)  # highest mean first

    # ---- 1b) default highlights: top 10 if none provided ----
    if not highlight_labels:
        highlight_labels = agg.index[:10].tolist()

    # ---- 2) plotting ----
    ranks = np.arange(1, len(agg) + 1)
    fig, ax = plt.subplots(figsize=figsize)

    # all in gray
    ax.errorbar(ranks, agg['mean'], yerr=agg['ci'],
                fmt='o', color='lightgray', ecolor='gray', capsize=4, zorder=1)

    # highlights in red
    texts = []
    for label in highlight_labels:
        if label not in agg.index:
            continue
        i    = agg.index.get_loc(label)
        y    = agg.iloc[i]['mean']
        yerr = agg.iloc[i]['ci']
        ax.errorbar(ranks[i], y, yerr=yerr,
                    fmt='o', color='red', capsize=4, zorder=2)
        txt = ax.text(ranks[i], y, f" {label}",
                      color='red', ha='left', va='bottom',
                      fontsize=9, zorder=3)
        texts.append(txt)

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    # ---- 3) final touches ----
    ax.set_xlabel('Rank (1 = highest mean)')
    ax.set_ylabel(value_col)
    ax.set_title(f'{value_col} by {group_col} (highlighted {len(texts)})')

    plt.tight_layout()
    plt.show()

    # return the full agg table and, if you like, the top-10 subset:
    top10 = agg.head(10)
    return agg, top10