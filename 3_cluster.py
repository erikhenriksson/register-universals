import argparse
import os
import pickle
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

# Set strict thread limits BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"  # Use absolute minimal threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Calculate clustering metrics on UMAP projections."
)
parser.add_argument(
    "--fold",
    type=int,
    required=True,
    help="Single fold number to process (1-based).",
)
parser.add_argument(
    "--N",
    type=int,
    default=1000,
    help="Number of samples per label per language (used for directory naming).",
)
parser.add_argument(
    "--ONLY_MAIN_LABEL",
    action="store_true",
    help="If set, process data from main_labels_only directory.",
)
parser.add_argument(
    "--random_state",
    type=int,
    default=42,
    help="Random state for KMeans for reproducibility.",
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="./",
    help="Base directory containing umap data directories.",
)
parser.add_argument(
    "--max_k",
    type=int,
    default=40,
    help="Maximum number of clusters to try for KMeans.",
)
parser.add_argument(
    "--min_k", type=int, default=2, help="Minimum number of clusters to try for KMeans."
)
parser.add_argument(
    "--step_k", type=int, default=1, help="Step size for k values in KMeans."
)
args = parser.parse_args()

# Setup directory paths
if args.ONLY_MAIN_LABEL:
    label_type = "main_labels_only"
else:
    label_type = "filtered_labels"

input_dir = os.path.join(args.base_dir, f"umap_data_{args.N}_{label_type}")
output_dir = os.path.join(args.base_dir, f"cluster_metrics_{args.N}_{label_type}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Define the range of k values
k_values = range(args.min_k, args.max_k + 1, args.step_k)


# Function to get the most common label for each sample
def get_majority_labels(metadata):
    labels = []
    for item in metadata:
        # Get only uppercase labels which are the main classes, exclude "MT"
        main_labels = [
            label for label in item["preds_0.4"] if label.isupper() and label != "MT"
        ]
        if main_labels:
            # Take the first label as the majority class
            labels.append(main_labels[0])
        else:
            # If no valid labels, use a placeholder
            labels.append("UNKNOWN")
    return np.array(labels)


# Function to calculate metrics for a specific embedding and k value
def calculate_metrics_for_k(data, true_labels, k, random_state):
    # Skip if number of samples is less than k
    if len(data) <= k:
        return {
            "k": k,
            "silhouette": None,
            "ari": None,
            "nmi": None,
            "v_measure": None,
            "homogeneity": None,
            "completeness": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

    try:
        # Fit KMeans with absolute minimal threading - single init, full algorithm
        kmeans = KMeans(
            n_clusters=k, random_state=random_state, n_init=1, algorithm="full"
        )
        cluster_labels = kmeans.fit_predict(data)

        # Calculate metrics
        sil_score = silhouette_score(data, cluster_labels)
        ari_score = adjusted_rand_score(true_labels, cluster_labels)
        nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
        v_score = v_measure_score(true_labels, cluster_labels)
        homo_score = homogeneity_score(true_labels, cluster_labels)
        comp_score = completeness_score(true_labels, cluster_labels)
        db_score = davies_bouldin_score(data, cluster_labels)
        ch_score = calinski_harabasz_score(data, cluster_labels)

        return {
            "k": k,
            "silhouette": sil_score,
            "ari": ari_score,
            "nmi": nmi_score,
            "v_measure": v_score,
            "homogeneity": homo_score,
            "completeness": comp_score,
            "davies_bouldin": db_score,
            "calinski_harabasz": ch_score,
        }
    except Exception as e:
        print(f"Error in KMeans for k={k}: {str(e)}")
        # Return None values for metrics on error
        return {
            "k": k,
            "silhouette": None,
            "ari": None,
            "nmi": None,
            "v_measure": None,
            "homogeneity": None,
            "completeness": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "error": str(e),
        }


# Main execution
def main():
    # Get just one fold number to process
    fold_number = args.fold
    fold_file = f"umap_fold_{fold_number}.pkl"
    fold_path = os.path.join(input_dir, fold_file)
    output_file = os.path.join(output_dir, f"metrics_fold_{fold_number}.pkl")

    print(f"Processing single fold: {fold_file}")
    print(
        f"Thread settings: OMP={os.environ['OMP_NUM_THREADS']}, OPENBLAS={os.environ['OPENBLAS_NUM_THREADS']}"
    )

    start_time = time.time()

    # Check if file exists
    if not os.path.exists(fold_path):
        print(f"Error: Fold file {fold_path} does not exist.")
        return

    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        return

    # Load the fold data
    with open(fold_path, "rb") as f:
        fold_data = pickle.load(f)

    metadata = fold_data["metadata"]
    embeddings = fold_data["embeddings"]

    print(f"Loaded {len(metadata)} samples from {fold_file}")

    # Get majority labels
    true_labels = get_majority_labels(metadata)

    # Process each embedding type and k value
    all_metrics = []

    # Define all embedding positions and types
    positions = ["first", "half", "last"]

    # Process each embedding position
    for position in positions:
        position_embeddings = embeddings[position]

        # Process raw embeddings
        print(f"Processing {position}_raw...")
        for k in k_values:
            metrics = calculate_metrics_for_k(
                position_embeddings["raw"], true_labels, k, args.random_state
            )
            metrics["embedding"] = f"{position}_raw"
            all_metrics.append(metrics)

        # Save intermediate results
        intermediate_file = os.path.join(
            output_dir, f"metrics_fold_{fold_number}_partial_{position}_raw.pkl"
        )
        with open(intermediate_file, "wb") as f:
            pickle.dump(all_metrics, f)
        print(f"Saved intermediate results to {intermediate_file}")

        # Process UMAP projections - one at a time
        for dim in [2, 4, 8, 16, 32]:
            umap_key = f"umap_{dim}d"
            print(f"Processing {position}_{umap_key}...")
            for k in k_values:
                metrics = calculate_metrics_for_k(
                    position_embeddings[umap_key], true_labels, k, args.random_state
                )
                metrics["embedding"] = f"{position}_{umap_key}"
                all_metrics.append(metrics)

            # Save intermediate results after each embedding type
            intermediate_file = os.path.join(
                output_dir,
                f"metrics_fold_{fold_number}_partial_{position}_{umap_key}.pkl",
            )
            with open(intermediate_file, "wb") as f:
                pickle.dump(all_metrics, f)
            print(f"Saved intermediate results to {intermediate_file}")

    # Save the final results
    with open(output_file, "wb") as f:
        pickle.dump(all_metrics, f)

    elapsed_time = time.time() - start_time
    print(
        f"Completed fold {fold_number} in {elapsed_time:.2f} seconds. Saved to {output_file}"
    )


if __name__ == "__main__":
    main()
