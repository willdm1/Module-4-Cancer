# main.py Module 4 cancer dataset
# Updated: 4/18/2026
# Written by Will and Dani

# Github sources we used: 
# Dr. Groves repository: https://github.com/smgroves/Module-4-Cancer
# scikit-learn repository: https://github.com/scikit-learn/scikit-learn
# UMAP repository: https://github.com/lmcinnes/umap
# pandas repository: https://github.com/pandas-dev/pandas

# Follows the 'example_EDA.py' format from our class

# We are loading the fils and exploring the data with pandas / scikit-learn
# %%p
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # type: ignore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap # type: ignore
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

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

# Here we look for the data folder in a couple of common capitalizations and take the first one that exists.
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if p.exists()), DATA_DIR_CANDIDATES[0])

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# We hardcode the paths to the training data and metadata CSVs, which are expected to be in the data folder.
TRAINING_DATA_PATH = DATA_DIR / "TRAINING_SET_GSE62944_subsample_log2TPM.csv"
TRAINING_METADATA_PATH = DATA_DIR / "TRAINING_SET_GSE62944_metadata.csv"
VALIDATION_DATA_PATH = DATA_DIR / "VALIDATION_SET_GSE62944_subsample_log2TPM.csv"
VALIDATION_METADATA_PATH = DATA_DIR / "VALIDATION_SET_GSE62944_metadata.csv"
NONNA_PATH = DATA_DIR / "GSE62944_metadata_percent_nonNA_by_cancer_type.csv"

# We are focusin this first check-in on one cancer type to reduce biological heterogeneity.
CANCER_TYPE = "COAD"  # Colon adenocarcinoma

# Hallmrk split for our team (20 genes per hallmark, 40 total):
# - Will: Susained proliferative signaling
# - Dani: Resisting apoptosis
PROLIFERATIVE_SIGNALING_GENES = [
    "EGFR", "ERBB2", "KRAS", "NRAS", "BRAF",
    "PIK3CA", "AKT1", "MYC", "FOS", "JUN",
    "MET", "IGF1R", "FGFR1", "FGFR2", "EGF",
    "HGF", "SOS1", "GRB2", "MAPK1", "MAPK3",
]

# We are using a mix of well-known oncogenes / tumor suppressors and some more general pathway members.
RESIST_APOPTOSIS_GENES = [
    "TP53", "BCL2", "BCL2L1", "MCL1", "CASP8",
    "CASP3", "BAX", "BAK1", "APAF1", "BID",
    "CASP9", "FAS", "FASLG", "BAD", "BBC3",
    "BCL2L11", "CFLAR", "DIABLO", "TNF", "TNFRSF10B",
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

def simplify_stage(stage_value: object) -> object:

    # Collapse detailed AJCC stae labels into broader Stage I / II / III / IV bins.
    # This makes the metadata easier to visualize and creates a cleaner future target for supervised learning.

    if pd.isna(stage_value): # type: ignore
        return pd.NA

    stage_string = str(stage_value).strip().upper()

    if stage_string in {"", "[NOT AVAILABLE]", "NAN"}:
        return pd.NA
    if "STAGE IV" in stage_string:
        return "Stage IV"
    if "STAGE III" in stage_string:
        return "Stage III"
    if "STAGE II" in stage_string:
        return "Stage II"
    if "STAGE I" in stage_string:
        return "Stage I"

    return pd.NA


def clean_cancer_metadata(cancer_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    # We clen metadata for downstream plotting and modeling.
    # Convert age to numeric if present
    # Fnd the stage column
    # Create simplified stage labels
    # Create a binary stage label for a future supervised classification model
    # (Early = Stage I/II, Late = Stage III/IV)
    cancer_metadata = cancer_metadata.copy()

    if "age_at_diagnosis" in cancer_metadata.columns:
        cancer_metadata["age_at_diagnosis"] = pd.to_numeric(
            cancer_metadata["age_at_diagnosis"],
            errors="coerce",
        )

    stage_col = find_existing_stage_column(cancer_metadata)

    # If we found a stage column, we create cleaned and simplified stage labels
    if stage_col is not None:
        cancer_metadata["stage_clean"] = cancer_metadata[stage_col].replace(
            "[Not Available]", pd.NA
        )
        cancer_metadata["stage_simple"] = cancer_metadata["stage_clean"].apply(simplify_stage) # type: ignore
        cancer_metadata["stage_binary"] = cancer_metadata["stage_simple"].map(
            {
                "Stage I": "Early",
                "Stage II": "Early",
                "Stage III": "Late",
                "Stage IV": "Late",
            }
        )

    return cancer_metadata, stage_col

def load_split_data(
    expression_path: Path,
    metadata_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not expression_path.exists():
        raise FileNotFoundError(f"Could not find expression matrix at: {expression_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata table at: {metadata_path}")

    data = pd.read_csv(expression_path, index_col=0, header=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0, header=0)
    return data, metadata_df


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_split_data(TRAINING_DATA_PATH, TRAINING_METADATA_PATH)

def summarize_expression_matrix(data: pd.DataFrame) -> pd.DataFrame:

    # Compue a concise summary of the full training expression matrix.
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

    # We find the samples with the specified cancer type in the metadata,
    # then subset both the metadata and expression matrix to those samples.
    cancer_samples = metadata_df.loc[metadata_df["cancer_type"] == cancer_type].index.tolist()
    cancer_samples = [sample for sample in cancer_samples if sample in data.columns]

    if len(cancer_samples) == 0:
        raise ValueError(
            f"No overlapping samples were found for cancer type '{cancer_type}'."
        )

    # We subset the expression matrix to the selected samples and align the metadata to those samples.
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

    # We check for the presence of each requested hallmark gene in the dataset 
    # build a table summarizing which genes were found.
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

    # We create a subset of the expression matrix with only the present hallmark genes for downstream analysis.
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

    # We check if any genes were present for each hallmark before computing the scores
    # if not we fill the score with NA values.
    if apoptosis_present:
        score_df["apoptosis_resistance_score"] = hallmark_gene_data.loc[apoptosis_present].mean(axis=0)
    else:
        score_df["apoptosis_resistance_score"] = pd.NA

    # We merge the scores with the cancer metadata so we have all sample-level information in one table for downstream plotting and modeling.
    merged = score_df.merge(cancer_metadata, left_index=True, right_index=True, how="left")
    return merged

def build_supervised_features(
    hallmark_gene_data: pd.DataFrame,
    merged_scores: pd.DataFrame,
    include_summary_features: bool = False,
) -> pd.DataFrame:
    # Start with raw hallmark-gene expression, samples x genes
    X = hallmark_gene_data.T.apply(pd.to_numeric, errors="coerce").copy()

    if include_summary_features:
        extra_cols = []

        if "proliferative_score" in merged_scores.columns:
            X["proliferative_score"] = pd.to_numeric(
                merged_scores.loc[X.index, "proliferative_score"], errors="coerce"
            )
            extra_cols.append("proliferative_score")

        if "apoptosis_resistance_score" in merged_scores.columns:
            X["apoptosis_resistance_score"] = pd.to_numeric(
                merged_scores.loc[X.index, "apoptosis_resistance_score"], errors="coerce"
            )
            extra_cols.append("apoptosis_resistance_score")

        if "age_at_diagnosis" in merged_scores.columns:
            X["age_at_diagnosis"] = pd.to_numeric(
                merged_scores.loc[X.index, "age_at_diagnosis"], errors="coerce"
            )
            extra_cols.append("age_at_diagnosis")

        # Fill any missing values in added summary features with training-column medians later
        # For now, leave NaN in place

    return X

def build_stage_binary_target(merged_scores: pd.DataFrame) -> pd.Series:
    y = merged_scores["stage_binary"].copy()
    y = y.map({"Early": 0, "Late": 1})
    return y

def align_feature_matrices(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_cols = [col for col in X_train.columns if col in X_valid.columns]
    X_train = X_train[common_cols].copy()
    X_valid = X_valid[common_cols].copy()
    return X_train, X_valid

def scale_train_valid(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    # Fill missing values using training medians only
    train_medians = X_train.median(axis=0)
    X_train_filled = X_train.fillna(train_medians)
    X_valid_filled = X_valid.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_valid_scaled = scaler.transform(X_valid_filled)

    return X_train_scaled, X_valid_scaled, scaler

def summarize_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    split_name: str,
    model_name: str,
) -> pd.DataFrame:
    metrics = {
        "model": model_name,
        "split": split_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
    }
    return pd.DataFrame([metrics])

def save_supervised_model_figures(
    baseline: Dict[str, object],
    improved: Dict[str, object],
) -> None:
   # 1) Validation ROC comparison
    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        baseline["y_valid"],
        baseline["valid_scores"],
        name="Baseline",
        ax=ax,
    )

    RocCurveDisplay.from_predictions(
        improved["y_valid"],
        improved["valid_scores"],
        name="Improved",
        ax=ax,
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"{CANCER_TYPE}: Validation ROC comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_validation_ROC_comparison.png", dpi=300)
    plt.show()
    plt.close()

    # 2) Validation confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ConfusionMatrixDisplay(
        confusion_matrix=baseline["valid_confusion_matrix"],
        display_labels=["Early", "Late"],
    ).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Baseline model")

    ConfusionMatrixDisplay(
        confusion_matrix=improved["valid_confusion_matrix"],
        display_labels=["Early", "Late"],
    ).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Improved model")

    fig.suptitle(f"{CANCER_TYPE}: Validation confusion matrices")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_validation_confusion_matrices.png", dpi=300)
    plt.show()
    plt.close()

def run_unsupervised_models(
    hallmark_gene_data: pd.DataFrame,
    merged_scores: pd.DataFrame,
) -> Dict[str, object]:

    # We build the sample-by-gene matrix used for PCA, UMAP, and KMeans clustering.

    # We follows the same sklearn workflow emphasized in lecture:
    # 1) build X = samples x features
    # 2) standardize the data
    # 3) fit PCA / UMAP / clustering models
    # 4) save coordinates and cluster assignments for plotting

    # sklearn expects samples x features, so transpose the gene matrix
    X = hallmark_gene_data.T.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=0, how="any")

    # Keep metadata aligned to the exact samples used in the model matrix
    model_metadata = merged_scores.loc[X.index].copy()

    # Standardize the gene-expression features before PCA / UMAP / KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        X_pca,
        index=X.index,
        columns=["PC1", "PC2"],
    )
    pca_df = pca_df.merge(model_metadata, left_index=True, right_index=True, how="left")

    # KMeans clustering
    clustering_rows = []
    best_k = None
    best_score = -1
    best_labels = None

    for k in range(2, 7):
        kmeans_model = KMeans(n_clusters=k, random_state=0, n_init=20)
        cluster_labels = kmeans_model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, cluster_labels)

        clustering_rows.append(
            {
                "k": k,
                "inertia": float(kmeans_model.inertia_),
                "silhouette_score": float(sil),
            }
        )

        if sil > best_score:
            best_score = sil
            best_k = k
            best_labels = cluster_labels

    cluster_eval_df = pd.DataFrame(clustering_rows)

    cluster_series = pd.Series(
        best_labels,
        index=X.index,
        name="kmeans_cluster",
    ).astype(str)

    # We add the cluster assignments to the PCA dataframe and the model metadata for downstream plotting and analysis.
    pca_df["kmeans_cluster"] = cluster_series
    model_metadata["kmeans_cluster"] = cluster_series

    # UMAP
    umap_df = None
    if HAS_UMAP:
        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )
        X_umap = umap_model.fit_transform(X_scaled)

        umap_df = pd.DataFrame(
            X_umap,
            index=X.index,
            columns=["UMAP1", "UMAP2"],
        )
        umap_df = umap_df.merge(model_metadata, left_index=True, right_index=True, how="left")

    return {
        "X": X,
        "pca_df": pca_df,
        "cluster_eval_df": cluster_eval_df,
        "best_k": best_k,
        "best_silhouette": best_score,
        "umap_df": umap_df,
    }

def make_summary_plots(
    hallmark_gene_data: pd.DataFrame,
    merged_scores: pd.DataFrame,
    stage_col: Optional[str],
    unsupervised_results: Dict[str, object],
) -> None:
  
    # Here is what we do:
    # 1) hallmark-score scatterplot
    # 2) hallmark-score boxplot by simplified stage
    # 3) PCA colored by stage
    # 4) PCA colored by KMeans cluster
    # 5) silhouette score vs. k for KMeans
    # 6) optional UMAP colored by stage
    # 7) optional UMAP colored by cluster
    # 8) DB SCAN Clustering 
  
    sns.set_theme(style="whitegrid")

    stage_plot_col = "stage_simple" if "stage_simple" in merged_scores.columns else stage_col

    # 1) Scatterplot of the two hallmark scores
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=merged_scores,
        x="proliferative_score",
        y="apoptosis_resistance_score",
        hue=stage_plot_col if stage_plot_col in merged_scores.columns else None,
        alpha=0.8,
    )
    plt.title(f"{CANCER_TYPE}: Hallmark score comparison")
    plt.xlabel("Mean proliferative-signaling expression")
    plt.ylabel("Mean apoptosis-resistance expression")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_score_scatter.png", dpi=300)
    plt.show()
    plt.close()

    # 2) Boxplot of hallmark scores by simplified stage
    if stage_plot_col in merged_scores.columns:
        stage_plot_df = merged_scores[
            [stage_plot_col, "proliferative_score", "apoptosis_resistance_score"]
        ].copy()
        stage_plot_df = stage_plot_df.dropna(subset=[stage_plot_col])

        if not stage_plot_df.empty:
            melted = stage_plot_df.melt(
                id_vars=stage_plot_col,
                value_vars=["proliferative_score", "apoptosis_resistance_score"],
                var_name="hallmark_score_type",
                value_name="score",
            )

            plt.figure(figsize=(9, 5))
            sns.boxplot(
                data=melted,
                x=stage_plot_col,
                y="score",
                hue="hallmark_score_type",
            )
            plt.title(f"{CANCER_TYPE}: Hallmark scores by simplified tumor stage")
            plt.xlabel("Simplified tumor stage")
            plt.ylabel("Mean hallmark expression score")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_scores_by_stage.png", dpi=300)
            plt.show()
            plt.close()

    # 3) PCA colored by stage
    pca_df = unsupervised_results["pca_df"]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=pca_df, # type: ignore
        x="PC1",
        y="PC2",
        hue=stage_plot_col if stage_plot_col in pca_df.columns else None, # type: ignore
        alpha=0.85,
    )
    plt.title(
        f"{CANCER_TYPE}: PCA of 40 hallmark genes"
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_PCA_by_stage.png", dpi=300)
    plt.show()
    plt.close()

    # 4) PCA colored by KMeans cluster
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=pca_df, # type: ignore
        x="PC1",
        y="PC2",
        hue="kmeans_cluster",
        palette="tab10",
        alpha=0.85,
    )
    plt.title(
        f"{CANCER_TYPE}: PCA colored by KMeans clusters "
        f"(best k = {unsupervised_results['best_k']})"
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_PCA_by_cluster.png", dpi=300)
    plt.show()
    plt.close()

    # 5) KMeans model-selection plot
    cluster_eval_df = unsupervised_results["cluster_eval_df"]

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=cluster_eval_df, # type: ignore
        x="k",
        y="silhouette_score",
        marker="o",
    )
    plt.axvline(
        x=unsupervised_results["best_k"], # type: ignore
        linestyle="--",
        color="black",
        label=f"best k = {unsupervised_results['best_k']}",
    )
    plt.title(f"{CANCER_TYPE}: KMeans silhouette score by k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_kmeans_silhouette.png", dpi=300)
    plt.show()
    plt.close()

    # 6/7) UMAP plots
    umap_df = unsupervised_results["umap_df"]

    if umap_df is not None:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=umap_df, # type: ignore
            x="UMAP1",
            y="UMAP2",
            hue=stage_plot_col if stage_plot_col in umap_df.columns else None, # type: ignore
            alpha=0.85,
        )
        plt.title(f"{CANCER_TYPE}: UMAP of 40 hallmark genes")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_UMAP_by_stage.png", dpi=300)
        plt.show()
        plt.close()

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=umap_df, # type: ignore
            x="UMAP1",
            y="UMAP2",
            hue="kmeans_cluster",
            palette="tab10",
            alpha=0.85,
        )
        plt.title(
            f"{CANCER_TYPE}: UMAP colored by KMeans clusters "
            f"(best k = {unsupervised_results['best_k']})"
        )
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{CANCER_TYPE}_UMAP_by_cluster.png", dpi=300)
        plt.show()
        plt.close()

def prepare_modeling_split(
    expression_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cancer_type: str,
) -> Dict[str, object]:
    cancer_data, cancer_metadata = subset_cancer_type(expression_df, metadata_df, cancer_type)
    cancer_metadata, stage_col = clean_cancer_metadata(cancer_metadata)

    hallmark_gene_data, hallmark_gene_presence = subset_hallmark_genes(cancer_data)
    if hallmark_gene_data.empty:
        raise ValueError("No requested hallmark genes were found in this split.")

    merged_scores = compute_hallmark_scores(hallmark_gene_data, cancer_metadata)

    return {
        "cancer_data": cancer_data,
        "cancer_metadata": cancer_metadata,
        "hallmark_gene_data": hallmark_gene_data,
        "hallmark_gene_presence": hallmark_gene_presence,
        "merged_scores": merged_scores,
        "stage_col": stage_col,
    }

def run_logistic_regression_experiment(
    train_processed: Dict[str, object],
    valid_processed: Dict[str, object],
    include_summary_features: bool,
    C_value: float,
    class_weight: Optional[str],
    model_name: str,
) -> Dict[str, object]:
    X_train = build_supervised_features(
        train_processed["hallmark_gene_data"],  # type: ignore
        train_processed["merged_scores"],       # type: ignore
        include_summary_features=include_summary_features,
    )
    X_valid = build_supervised_features(
        valid_processed["hallmark_gene_data"],  # type: ignore
        valid_processed["merged_scores"],       # type: ignore
        include_summary_features=include_summary_features,
    )

    y_train = build_stage_binary_target(train_processed["merged_scores"])  # type: ignore
    y_valid = build_stage_binary_target(valid_processed["merged_scores"])  # type: ignore

    # Keep only samples with known Early/Late labels
    train_keep = y_train.notna()
    valid_keep = y_valid.notna()

    X_train = X_train.loc[train_keep]
    X_valid = X_valid.loc[valid_keep]
    y_train = y_train.loc[train_keep].astype(int)
    y_valid = y_valid.loc[valid_keep].astype(int)

    X_train, X_valid = align_feature_matrices(X_train, X_valid)
    X_train_scaled, X_valid_scaled, scaler = scale_train_valid(X_train, X_valid)

    clf = LogisticRegression(
        C=C_value,
        class_weight=class_weight,
        max_iter=5000,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    train_score = clf.predict_proba(X_train_scaled)[:, 1]
    valid_score = clf.predict_proba(X_valid_scaled)[:, 1]

    train_pred = (train_score >= 0.5).astype(int)
    valid_pred = (valid_score >= 0.5).astype(int)

    train_metrics = summarize_classification_metrics(
        y_train, train_pred, train_score, "train", model_name
    )
    valid_metrics = summarize_classification_metrics(
        y_valid, valid_pred, valid_score, "validation", model_name
    )

    train_cm = confusion_matrix(y_train, train_pred)
    valid_cm = confusion_matrix(y_valid, valid_pred)

    feature_names = X_train.columns.tolist()
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": clf.coef_[0],
        "abs_coefficient": np.abs(clf.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)

    return {
        "model_name": model_name,
        "classifier": clf,
        "scaler": scaler,
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "all_metrics": pd.concat([train_metrics, valid_metrics], ignore_index=True),
        "train_confusion_matrix": train_cm,
        "valid_confusion_matrix": valid_cm,
        "train_scores": train_score,
        "valid_scores": valid_score,
        "train_preds": train_pred,
        "valid_preds": valid_pred,
        "coef_df": coef_df,
    }


def run_supervised_modeling() -> Dict[str, object]:
    train_expr, train_meta = load_split_data(TRAINING_DATA_PATH, TRAINING_METADATA_PATH)
    valid_expr, valid_meta = load_split_data(VALIDATION_DATA_PATH, VALIDATION_METADATA_PATH)

    train_processed = prepare_modeling_split(train_expr, train_meta, CANCER_TYPE)
    valid_processed = prepare_modeling_split(valid_expr, valid_meta, CANCER_TYPE)

    # Baseline model:
    # raw 40 hallmark genes only, default class weighting
    baseline = run_logistic_regression_experiment(
        train_processed=train_processed,
        valid_processed=valid_processed,
        include_summary_features=False,
        C_value=1.0,
        class_weight=None,
        model_name="baseline_logreg_40genes",
    )

    # Improved model:
    # raw 40 genes + hallmark summary scores + age, with balanced classes and stronger regularization
    improved = run_logistic_regression_experiment(
        train_processed=train_processed,
        valid_processed=valid_processed,
        include_summary_features=True,
        C_value=0.1,
        class_weight="balanced",
        model_name="improved_logreg_40genes_plus_scores_age_balanced",
    )

    comparison_df = pd.concat(
        [baseline["all_metrics"], improved["all_metrics"]],
        ignore_index=True,
    )

    comparison_df.to_csv(
        RESULTS_DIR / f"{CANCER_TYPE}_supervised_model_metrics.csv",
        index=False,
    )
    baseline["coef_df"].to_csv(
        RESULTS_DIR / f"{CANCER_TYPE}_baseline_logreg_coefficients.csv",
        index=False,
    )
    improved["coef_df"].to_csv(
        RESULTS_DIR / f"{CANCER_TYPE}_improved_logreg_coefficients.csv",
        index=False,
    )

    save_supervised_model_figures(baseline, improved)

    return {
        "train_processed": train_processed,
        "valid_processed": valid_processed,
        "baseline": baseline,
        "improved": improved,
        "comparison_df": comparison_df,
    }

    # %%
# Main analysis workflow
####################################################
def run_eda() -> Dict[str, object]:

    # We run the complete EDA workflow for the Module 4 check-in.
    # Returns all major tables so the notebook can reuse them directly.

    data, metadata_df = load_data()

    # We print out summaries and sanity checks at each step to verify that the data looks as expected 
    # highlight key characteristics of the dataset.
    print("Loaded expression data.")
    print(data.head())
    print("\nExpression matrix shape (genes x samples):", data.shape)
    print("\nMetadata shape (samples x features):", metadata_df.shape)

    print("\nExpression matrix info:")
    print(data.info())

    print("\nMetadata info:")
    print(metadata_df.info())

    # We compute a global summary of the expression matrix to understand the overall distribution of expression values
    #  identify any potential issues (e.g. all zeros, unexpected ranges).
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

    # We assess the missingness in the metadata to understand which features are well-populated and which have a lot of missing values.
    missingness_df = report_missingness(metadata_df)
    print("\nMetadata missingness summary:")
    print(missingness_df.head(20))

    # We subset the data to the selected cancer type and verify that the subsetting worked correctly
    # checking the shapes and printing out some of the metadata.
    cancer_data, cancer_metadata = subset_cancer_type(data, metadata_df, CANCER_TYPE)
    print(f"\nSubset to {CANCER_TYPE}:")
    print("Expression shape (genes x selected samples):", cancer_data.shape)
    print("Metadata shape (selected samples x features):", cancer_metadata.shape)

    # We clean the metadata for the selected cancer type, which includes finding the stage column, creating simplified stage labels
    # creating a binary stage label for future modeling.
    cancer_metadata, stage_col = clean_cancer_metadata(cancer_metadata)

    if "stage_simple" in cancer_metadata.columns:
        print("\nSimplified stage counts:")
        print(cancer_metadata["stage_simple"].value_counts(dropna=False))

    if "stage_binary" in cancer_metadata.columns:
        print("\nBinary stage counts (planned future supervised target):")
        print(cancer_metadata["stage_binary"].value_counts(dropna=False))

    hallmark_gene_data, hallmark_gene_presence = subset_hallmark_genes(cancer_data)
    print("\nRequested hallmark genes and whether they were found:")
    print(hallmark_gene_presence)

    # We check if any of the requested hallmark genes were present in the dataset before proceeding with downstream analysis.
    if hallmark_gene_data.empty:
        raise ValueError(
            "None of the requested hallmark genes were found in the dataset. "
            "Update the gene lists and rerun."
        )

    print("\nHallmark gene expression subset:")
    print(hallmark_gene_data.head())

    # We compute summary statistics for the hallmark genes to understand their expression distribution within the selected cancer type.
    hallmark_gene_stats = pd.DataFrame({
        "mean_expression": hallmark_gene_data.mean(axis=1),
        "median_expression": hallmark_gene_data.median(axis=1),
        "variance_across_samples": hallmark_gene_data.var(axis=1),
    }).sort_values("mean_expression", ascending=False)

    print("\nHallmark gene statistics within the selected cancer type:")
    print(hallmark_gene_stats)

    # We compute the hallmark scores for each sample and merge them with the metadata for downstream analysis and plotting.
    if stage_col is not None:
        print(f"\nUsing metadata stage column: {stage_col}")
        print(cancer_metadata[stage_col].value_counts(dropna=False))
    else:
        print("\nNo stage column was found among the expected metadata names.")

    merged_scores = compute_hallmark_scores(hallmark_gene_data, cancer_metadata)
    print("\nSample-level hallmark score table:")
    print(merged_scores.head())

    # We run unsupervised models (PCA, KMeans, optional UMAP) on the hallmark gene expression data to explore potential clusters and patterns in the data.
    unsupervised_results = run_unsupervised_models(hallmark_gene_data, merged_scores)

    print("\nKMeans model selection summary:")
    print(unsupervised_results["cluster_eval_df"])

    print(
        f"\nBest KMeans k based on silhouette score: "
        f"{unsupervised_results['best_k']} "
        f"(silhouette = {unsupervised_results['best_silhouette']:.3f})"
    )

    # Save the most important outputs for the notebook/report
    full_summary_df.to_csv(RESULTS_DIR / "training_expression_summary.csv", index=False)
    top_mean_expression.to_csv(RESULTS_DIR / "top20_mean_expression_genes.csv")
    missingness_df.to_csv(RESULTS_DIR / "metadata_missingness_summary.csv")
    hallmark_gene_presence.to_csv(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_gene_presence.csv", index=False)
    hallmark_gene_stats.to_csv(RESULTS_DIR / f"{CANCER_TYPE}_hallmark_gene_stats.csv")
    merged_scores.to_csv(RESULTS_DIR / f"{CANCER_TYPE}_sample_level_hallmark_scores.csv")

    # We save the cluster evaluation metrics and the PCA / UMAP coordinates with cluster assignments for downstream plotting and analysis in the notebook.
    unsupervised_results["cluster_eval_df"].to_csv( # type: ignore
        RESULTS_DIR / f"{CANCER_TYPE}_kmeans_cluster_evaluation.csv",
        index=False,
    )
    unsupervised_results["pca_df"].to_csv( # type: ignore
        RESULTS_DIR / f"{CANCER_TYPE}_PCA_coordinates_with_clusters.csv"
    )

    # We check if UMAP results are available before trying to save them, since UMAP is an optional dependency.
    if unsupervised_results["umap_df"] is not None:
        unsupervised_results["umap_df"].to_csv( # type: ignore
            RESULTS_DIR / f"{CANCER_TYPE}_UMAP_coordinates_with_clusters.csv"
        )

    # Save metadata completeness file 
    if NONNA_PATH.exists():
        nonna_df = pd.read_csv(NONNA_PATH)
        nonna_df.to_csv(RESULTS_DIR / "metadata_percent_nonNA_by_cancer_type_copy.csv", index=False)
        print("\nCopied metadata completeness table to results folder.")

    # Finally, we create and save summary plots to visualize the hallmark scores, PCA clusters, and KMeans model selection results.
    make_summary_plots(hallmark_gene_data, merged_scores, stage_col, unsupervised_results,)
    print("\nEDA complete. Results were saved to:")
    print(RESULTS_DIR)

    # We return all the major tables and results as a dictionary so they can be easily reused in the notebook without needing to reload or recompute them.
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
        "cluster_eval": unsupervised_results["cluster_eval_df"],
        "pca_df": unsupervised_results["pca_df"],
        "umap_df": unsupervised_results["umap_df"],      
    }

# %%
# Script entry point
####################################################
# We wrap the entire EDA workflow in a function and call it in the main block. 
# This allows us to easily reuse the function in the notebook and also keeps the script organized.
if __name__ == "__main__":
    eda_results = run_eda()

    supervised_results = run_supervised_modeling()
    print("\nSupervised modeling comparison:")
    print(supervised_results["comparison_df"])
