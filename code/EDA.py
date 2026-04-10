# Exploratory data analysis (EDA) on the Module 4 cancer dataset
# Updated: 4/9/2026
# Written by Will and Dani


# We are loading the files and exploring the data with pandas / scikit-learn
# %%p
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Project paths and analysis settings
####################################################
# This script is written to live inside our "code" foldr of our Module-4-Cancer repo.
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "Data",
]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if p.exists()), DATA_DIR_CANDIDATES[0])

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DATA_PATH = DATA_DIR / "TRAINING_SET_GSE62944_subsample_log2TPM.csv"
TRAINING_METADATA_PATH = DATA_DIR / "TRAINING_SET_GSE62944_metadata.csv"
NONNA_PATH = DATA_DIR / "GSE62944_metadata_percent_nonNA_by_cancer_type.csv"

# We are focusing this first check-in on one cancer type to reduce biological heterogeneity.
CANCER_TYPE = "COAD"  # Colon adenocarcinoma

# Hallmrk split for our team:
# - Will: Sustained proliferative signaling
# - Dani: Resisting apoptosis
PROLIFERATIVE_SIGNALING_GENES = [
    "EGFR", "ERBB2", "KRAS", "NRAS", "BRAF",
    "PIK3CA", "AKT1", "MYC", "FOS", "JUN",
]

RESIST_APOPTOSIS_GENES = [
    "TP53", "BCL2", "BCL2L1", "MCL1", "CASP8",
    "CASP3", "BAX", "BAK1", "APAF1", "BID",
]

# %%
# Helper functions
####################################################
def find_existing_stage_column(df: pd.DataFrame) -> Optional[str]:

    # We try several likely metadata column names for tumor stage and return the first one that exists.
    candidate_columns = [
        "ajcc_pathologic_tumor_stage",
        "pathologic_stage",
        "tumor_stage",
        "clinical_stage",
        "ajcc_pathologic_tumor_stage_short",
        "summary_stage",
        "stage_event_pathologic_stage",
    ]
    for col in candidate_columns:
        if col in df.columns:
            return col
    return None


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Load the training expression matrix and the training metadata table.
    # Expression data are expected to be genes x samples with log2(TPM+1) values.
    # Metadata are expected to be samples x features.
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find expression matrix at: {TRAINING_DATA_PATH}\n"
            "Check that the training CSV is present in the repo data folder."
        )
    if not TRAINING_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find metadata table at: {TRAINING_METADATA_PATH}\n"
            "Check that the metadata CSV is present in the repo data folder."
        )

    data = pd.read_csv(TRAINING_DATA_PATH, index_col=0, header=0)
    metadata_df = pd.read_csv(TRAINING_METADATA_PATH, index_col=0, header=0)
    return data, metadata_df


def summarize_expression_matrix(data: pd.DataFrame) -> pd.DataFrame:

    # Compute a concise summary of the full training expression matrix.
    # This helps answer the min/max/mean expression and the genes with the highest average expression.

    flattened = data.to_numpy().ravel()
    summary_dict = {
        "n_genes": [data.shape[0]],
        "n_samples": [data.shape[1]],
        "global_min_log2TPMplus1": [float(data.min().min())],
        "global_max_log2TPMplus1": [float(data.max().max())],
        "global_mean_log2TPMplus1": [float(flattened.mean())],
        "global_median_log2TPMplus1": [float(pd.Series(flattened).median())],
    }
    summary_df = pd.DataFrame(summary_dict)
    return summary_df


def report_missingness(metadata_df: pd.DataFrame) -> pd.DataFrame:
    
    # We calculate percent non-missing / missing for every metadata column.
    # This is useful because the lectures explicitly note that metadata completeness varies by cancer type.
    
    missing_df = pd.DataFrame({
        "n_non_missing": metadata_df.notna().sum(),
        "n_missing": metadata_df.isna().sum(),
        "percent_non_missing": 100 * metadata_df.notna().mean(),
        "percent_missing": 100 * metadata_df.isna().mean(),
    }).sort_values("percent_non_missing", ascending=False)
    return missing_df


def subset_cancer_type(data: pd.DataFrame, metadata_df: pd.DataFrame, cancer_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # We Subset both the expression matrix and metadata to one cancer type.
    
    if "cancer_type" not in metadata_df.columns:
        raise KeyError(
            "The metadata file does not contain a 'cancer_type' column, "
            "so the dataset cannot be subset by cancer type."
        )

    cancer_samples = metadata_df.loc[metadata_df["cancer_type"] == cancer_type].index.tolist()
    cancer_samples = [sample for sample in cancer_samples if sample in data.columns]

    if len(cancer_samples) == 0:
        raise ValueError(
            f"No overlapping samples were found for cancer type '{cancer_type}'."
        )

    cancer_data = data[cancer_samples]
    cancer_metadata = metadata_df.loc[cancer_samples].copy()
    return cancer_data, cancer_metadata


def subset_hallmark_genes(cancer_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Keep only the hallmark genes of interest and record which requested genes were actually found in the local dataset.

    requested_rows = []
    all_requested = (
        [("Sustained_Proliferative_Signaling", g) for g in PROLIFERATIVE_SIGNALING_GENES] +
        [("Resist_Apoptosis", g) for g in RESIST_APOPTOSIS_GENES]
    )

    present_genes: List[str] = []
    for hallmark_name, gene in all_requested:
        present = gene in cancer_data.index
        requested_rows.append({
            "hallmark": hallmark_name,
            "gene": gene,
            "present_in_dataset": present,
        })
        if present:
            present_genes.append(gene)

    requested_df = pd.DataFrame(requested_rows)
    unique_present_genes = list(dict.fromkeys(present_genes))
    subset_df = cancer_data.loc[unique_present_genes].copy() if unique_present_genes else pd.DataFrame()

    return subset_df, requested_df


def compute_hallmark_scores(
    hallmark_gene_data: pd.DataFrame,
    cancer_metadata: pd.DataFrame,
) -> pd.DataFrame:
    
    # Create per-sample hallmark scores by taking the mean expression across the
    # genes available for each hallmark. 
    # This is a simple first-pass signature

    proliferative_present = [g for g in PROLIFERATIVE_SIGNALING_GENES if g in hallmark_gene_data.index]
    apoptosis_present = [g for g in RESIST_APOPTOSIS_GENES if g in hallmark_gene_data.index]

    score_df = pd.DataFrame(index=hallmark_gene_data.columns)
    if proliferative_present:
        score_df["proliferative_score"] = hallmark_gene_data.loc[proliferative_present].mean(axis=0)
    else:
        score_df["proliferative_score"] = pd.NA

    if apoptosis_present:
        score_df["apoptosis_resistance_score"] = hallmark_gene_data.loc[apoptosis_present].mean(axis=0)
    else:
        score_df["apoptosis_resistance_score"] = pd.NA

    merged = score_df.merge(cancer_metadata, left_index=True, right_index=True, how="left")
    return merged


def make_summary_plots(
    hallmark_gene_data: pd.DataFrame,
    merged_scores: pd.DataFrame,
    stage_col: Optional[str],
) -> None:
    
    # Generate a small set of EDA figures for the notebook:
    # 1) scatterplot comparing hallmark scores
    # 2) boxplot by stage (if a stage column exists)
    # 3) PCA plot across the hallmark gene expression matrix

    sns.set_theme(style="whitegrid")

    # 1) Scatterplot of hallmark scores
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=merged_scores,
        x="proliferative_score",
        y="apoptosis_resistance_score",
        hue=stage_col if stage_col is not None else None,
        alpha=0.8,
    )
    plt.title(f"{CANCER_TYPE}: Hallmark score comparison")
    plt.xlabel("Mean proliferative-signaling expression")
    plt.ylabel("Mean apoptosis-resistance expression")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_score_scatter.png", dpi=300)
    plt.show()
    plt.close()

    # 2) Boxplot of hallmark score by stage (only if a stage column exists)
    if stage_col is not None:
        stage_plot_df = merged_scores[[stage_col, "proliferative_score", "apoptosis_resistance_score"]].copy()
        stage_plot_df = stage_plot_df.dropna(subset=[stage_col])

        if not stage_plot_df.empty:
            melted = stage_plot_df.melt(
                id_vars=stage_col,
                value_vars=["proliferative_score", "apoptosis_resistance_score"],
                var_name="hallmark_score_type",
                value_name="score",
            )

            plt.figure(figsize=(10, 5))
            sns.boxplot(data=melted, x=stage_col, y="score", hue="hallmark_score_type")
            plt.title(f"{CANCER_TYPE}: Hallmark scores by tumor stage")
            plt.xlabel(stage_col)
            plt.ylabel("Mean hallmark expression score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_scores_by_stage.png", dpi=300)
            plt.show()
            plt.close()

    # 3) PCA on hallmark genes using scikit-learn
    if hallmark_gene_data.shape[0] >= 2 and hallmark_gene_data.shape[1] >= 3:
        # PCA expects samples x features, so we transpose first
        X = hallmark_gene_data.T
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=0)
        pcs = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(
            pcs,
            index=X.index,
            columns=["PC1", "PC2"],
        )
        pca_df = pca_df.merge(merged_scores, left_index=True, right_index=True, how="left")

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=stage_col if stage_col is not None else None,
            alpha=0.85,
        )
        plt.title(
            f"{CANCER_TYPE}: PCA of hallmark genes\n"
            f"PC1={pca.explained_variance_ratio_[0]:.2%}, "
            f"PC2={pca.explained_variance_ratio_[1]:.2%}"
        )
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_gene_PCA.png", dpi=300)
        plt.show()
        plt.close()

        pca_df.to_csv(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_gene_PCA_coordinates.csv")


# %%
# Main analysis workflow
####################################################
def run_eda() -> Dict[str, pd.DataFrame]:

    # We run the complete EDA workflow for the Module 4 check-in.
    # Returns all major tables so the notebook can reuse them directly.

    data, metadata_df = load_data()

    print("Loaded expression data.")
    print(data.head())
    print("\nExpression matrix shape (genes x samples):", data.shape)
    print("\nMetadata shape (samples x features):", metadata_df.shape)

    print("\nExpression matrix info:")
    print(data.info())

    print("\nMetadata info:")
    print(metadata_df.info())

    full_summary_df = summarize_expression_matrix(data)
    print("\nGlobal expression summary:")
    print(full_summary_df)

    top_mean_expression = (
        data.mean(axis=1)
        .sort_values(ascending=False)
        .head(20)
        .rename("mean_expression")
        .to_frame()
    )
    print("\nTop 20 genes by average expression across the training set:")
    print(top_mean_expression)

    missingness_df = report_missingness(metadata_df)
    print("\nMetadata missingness summary:")
    print(missingness_df.head(20))

    cancer_data, cancer_metadata = subset_cancer_type(data, metadata_df, CANCER_TYPE)
    print(f"\nSubset to {CANCER_TYPE}:")
    print("Expression shape (genes x selected samples):", cancer_data.shape)
    print("Metadata shape (selected samples x features):", cancer_metadata.shape)

    hallmark_gene_data, hallmark_gene_presence = subset_hallmark_genes(cancer_data)
    print("\nRequested hallmark genes and whether they were found:")
    print(hallmark_gene_presence)

    if hallmark_gene_data.empty:
        raise ValueError(
            "None of the requested hallmark genes were found in the dataset. "
            "Update the gene lists and rerun."
        )

    print("\nHallmark gene expression subset:")
    print(hallmark_gene_data.head())

    hallmark_gene_stats = pd.DataFrame({
        "mean_expression": hallmark_gene_data.mean(axis=1),
        "median_expression": hallmark_gene_data.median(axis=1),
        "variance_across_samples": hallmark_gene_data.var(axis=1),
    }).sort_values("mean_expression", ascending=False)

    print("\nHallmark gene statistics within the selected cancer type:")
    print(hallmark_gene_stats)

    stage_col = find_existing_stage_column(cancer_metadata)
    if stage_col is not None:
        print(f"\nUsing metadata stage column: {stage_col}")
        print(cancer_metadata[stage_col].value_counts(dropna=False))
    else:
        print("\nNo stage column was found among the expected metadata names.")

    merged_scores = compute_hallmark_scores(hallmark_gene_data, cancer_metadata)
    print("\nSample-level hallmark score table:")
    print(merged_scores.head())

    # Save the most important outputs for the notebook/report
    full_summary_df.to_csv(RESULTS_DIR / "training_expression_summary.csv", index=False)
    top_mean_expression.to_csv(RESULTS_DIR / "top20_mean_expression_genes.csv")
    missingness_df.to_csv(RESULTS_DIR / "metadata_missingness_summary.csv")
    hallmark_gene_presence.to_csv(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_gene_presence.csv", index=False)
    hallmark_gene_stats.to_csv(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_gene_stats.csv")
    merged_scores.to_csv(RESULTS_DIR / f"{CANCER_TYPE}_sample_level_hallmark_scores.csv")

    # Save metadata completeness file 
    if NONNA_PATH.exists():
        nonna_df = pd.read_csv(NONNA_PATH)
        nonna_df.to_csv(RESULTS_DIR / "metadata_percent_nonNA_by_cancer_type_copy.csv", index=False)
        print("\nCopied metadata completeness table to results folder.")

    make_summary_plots(hallmark_gene_data, merged_scores, stage_col)

    print("\nEDA complete. Results were saved to:")
    print(RESULTS_DIR)

    return {
        "data": data,
        "metadata": metadata_df,
        "full_summary": full_summary_df,
        "top_mean_expression": top_mean_expression,
        "missingness": missingness_df,
        "cancer_data": cancer_data,
        "cancer_metadata": cancer_metadata,
        "hallmark_gene_data": hallmark_gene_data,
        "hallmark_gene_presence": hallmark_gene_presence,
        "hallmark_gene_stats": hallmark_gene_stats,
        "merged_scores": merged_scores,
    }

# %%
# Script entry point
####################################################
if __name__ == "__main__":
    run_eda()
