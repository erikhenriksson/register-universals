import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Calculate clustering metrics on UMAP projections."
)
parser.add_argument(
    "--max_workers",
    type=int,
    default=10,
    help="Number of fold-level parallel processes to use.",
)
parser.add_argument(
    "--threads_per_worker",
    type=int,
    default=4,
    help="Number of threads for each worker to use.",
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

    # Fit KMeans - compatible with scikit-learn 1.6.1
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
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


# Worker initialization function to set thread limits
def init_worker(threads):
    # Set threading environment variables for this worker process
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)


# Function to process a single fold file
def process_fold(fold_file, input_dir, output_dir, k_values, random_state):
    start_time = time.time()
    fold_name = os.path.basename(fold_file)
    fold_number = fold_name.split("_")[2].split(".")[0]

    print(f"Processing {fold_name}...")

    # Load the fold data
    fold_path = os.path.join(input_dir, fold_file)
    with open(fold_path, "rb") as f:
        fold_data = pickle.load(f)

    metadata = fold_data["metadata"]
    embeddings = fold_data["embeddings"]

    print(f"  Loaded {len(metadata)} samples from {fold_name}")

    # Get majority labels
    true_labels = get_majority_labels(metadata)

    # Process each embedding type and k value
    all_metrics = []

    # Define all embedding positions and types we want to process
    positions = ["first", "half", "last"]
    embedding_types = ["raw"] + [f"umap_{dim}d" for dim in [2, 4, 8, 16, 32]]

    total_embeddings = len(positions) * len(embedding_types)
    print(f"  Processing {total_embeddings} embedding types sequentially...")

    # Process each embedding type
    for position in positions:
        position_embeddings = embeddings[position]

        # Process raw embeddings
        print(f"  Processing {position}_raw...")
        for k in k_values:
            metrics = calculate_metrics_for_k(
                position_embeddings["raw"], true_labels, k, random_state
            )
            metrics["embedding"] = f"{position}_raw"
            all_metrics.append(metrics)

        # Process UMAP projections
        for dim in [2, 4, 8, 16, 32]:
            umap_key = f"umap_{dim}d"
            print(f"  Processing {position}_{umap_key}...")
            for k in k_values:
                metrics = calculate_metrics_for_k(
                    position_embeddings[umap_key], true_labels, k, random_state
                )
                metrics["embedding"] = f"{position}_{umap_key}"
                all_metrics.append(metrics)

    # Save the results
    output_file = os.path.join(output_dir, f"metrics_fold_{fold_number}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_metrics, f)

    # Create a summary of the results
    summary = {
        "fold": fold_number,
        "num_samples": len(metadata),
        "unique_labels": len(np.unique(true_labels)),
        "label_distribution": {
            label: np.sum(true_labels == label) for label in np.unique(true_labels)
        },
    }

    elapsed_time = time.time() - start_time
    print(
        f"  Completed {fold_name} in {elapsed_time:.2f} seconds. Saved to {output_file}"
    )

    return fold_number, summary


# Main execution
def main():
    print(f"Starting clustering metrics calculation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.max_workers} parallel workers for folds")
    print(f"Using {args.threads_per_worker} threads per worker")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # List fold files
    fold_files = [
        f
        for f in os.listdir(input_dir)
        if f.startswith("umap_fold_") and f.endswith(".pkl")
    ]
    fold_files.sort()

    if not fold_files:
        print(f"Error: No fold files found in {input_dir}")
        return

    print(f"Found {len(fold_files)} fold files to process")
    print(
        f"Running KMeans with k values from {args.min_k} to {args.max_k} (step {args.step_k})"
    )

    # Set up the process pool and run the processing
    process_fold_partial = partial(
        process_fold,
        input_dir=input_dir,
        output_dir=output_dir,
        k_values=k_values,
        random_state=args.random_state,
    )

    # Process folds in parallel
    num_workers = min(
        args.max_workers, len(fold_files)
    )  # Don't use more workers than files
    print(f"Processing folds with {num_workers} parallel workers")

    if num_workers > 1:
        # Use the initializer to set thread limits for each worker process
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(args.threads_per_worker,),
        ) as executor:
            # Using list() to ensure we wait for all tasks to complete
            results = list(executor.map(process_fold_partial, fold_files))
    else:
        # For single worker, set thread limits in the main process
        init_worker(args.threads_per_worker)
        print("Processing folds sequentially")
        results = []
        for fold_file in tqdm(fold_files, desc="Processing folds"):
            results.append(process_fold_partial(fold_file))

    # Save a combined summary of all folds
    summaries = {fold_number: summary for fold_number, summary in results}
    summary_file = os.path.join(output_dir, "fold_summaries.pkl")
    with open(summary_file, "wb") as f:
        pickle.dump(summaries, f)

    # Print summary
    print("\nProcessing summary:")
    for fold_number, summary in sorted(results, key=lambda x: x[0]):
        print(
            f"  Fold {fold_number}: {summary['num_samples']} samples, {summary['unique_labels']} unique labels"
        )

    print(f"\nClustering metrics calculation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
