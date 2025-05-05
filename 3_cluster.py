import argparse
import concurrent.futures
import multiprocessing as mp
import os
import pickle
import time
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
    "--cpu_fraction",
    type=float,
    default=0.9,
    help="Fraction of CPUs to use for parallel processing (0.0-1.0).",
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
parser.add_argument(
    "--max_parallel_folds",
    type=int,
    default=8,
    help="Maximum number of folds to process in parallel",
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

# Automatically determine number of worker processes
available_cpus = mp.cpu_count()
num_workers = max(1, int(available_cpus * args.cpu_fraction))
print(
    f"Detected {available_cpus} CPUs, using {num_workers} worker processes ({args.cpu_fraction * 100:.0f}%)"
)

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
            # Take the first label as the majority class (since we already filtered in the sampling script)
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

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(data)

    # Calculate metrics with specific exception handling
    try:
        sil_score = silhouette_score(data, cluster_labels)
    except ValueError as e:
        print(f"Silhouette score calculation failed for k={k}: {e}")
        sil_score = None

    try:
        ari_score = adjusted_rand_score(true_labels, cluster_labels)
    except ValueError as e:
        print(f"ARI score calculation failed for k={k}: {e}")
        ari_score = None

    try:
        nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
    except ValueError as e:
        print(f"NMI score calculation failed for k={k}: {e}")
        nmi_score = None

    try:
        v_score = v_measure_score(true_labels, cluster_labels)
    except ValueError as e:
        print(f"V-measure calculation failed for k={k}: {e}")
        v_score = None

    try:
        homo_score = homogeneity_score(true_labels, cluster_labels)
    except ValueError as e:
        print(f"Homogeneity score calculation failed for k={k}: {e}")
        homo_score = None

    try:
        comp_score = completeness_score(true_labels, cluster_labels)
    except ValueError as e:
        print(f"Completeness score calculation failed for k={k}: {e}")
        comp_score = None

    try:
        db_score = davies_bouldin_score(data, cluster_labels)
    except ValueError as e:
        print(f"Davies-Bouldin score calculation failed for k={k}: {e}")
        db_score = None

    try:
        ch_score = calinski_harabasz_score(data, cluster_labels)
    except ValueError as e:
        print(f"Calinski-Harabasz score calculation failed for k={k}: {e}")
        ch_score = None

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


# Function to process a single embedding type with all k values
def process_embedding_with_k_values(task_data):
    embedding_name, data, true_labels, k_values, random_state = task_data
    results = []

    for k in k_values:
        metrics = calculate_metrics_for_k(data, true_labels, k, random_state)
        metrics["embedding"] = embedding_name
        results.append(metrics)

    return results


# Function to process embeddings (UMAP or raw)
def process_embeddings(embedding_name, embeddings, true_labels, k_values, random_state):
    results = []

    for k in k_values:
        metrics = calculate_metrics_for_k(embeddings, true_labels, k, random_state)
        metrics["embedding"] = embedding_name
        results.append(metrics)

    return results


# Function to process a single fold file with parallelization for embeddings
def process_fold_with_parallelism(args):
    fold_file, input_dir, output_dir, k_values, random_state, num_cpus = args
    start_time = time.time()
    fold_name = os.path.basename(fold_file)
    fold_number = fold_name.split("_")[2].split(".")[0]

    print(f"Processing {fold_name} with {num_cpus} CPUs...")

    # Load the fold data
    fold_path = os.path.join(input_dir, fold_file)
    with open(fold_path, "rb") as f:
        fold_data = pickle.load(f)

    metadata = fold_data["metadata"]
    embeddings = fold_data["embeddings"]

    print(f"  Loaded {len(metadata)} samples from {fold_name}")

    # Get majority labels
    true_labels = get_majority_labels(metadata)

    # Create a list of all embedding processing tasks
    tasks = []
    for position in ["first", "half", "last"]:
        position_embeddings = embeddings[position]
        # Add raw embeddings task
        tasks.append(
            (
                f"{position}_raw",
                position_embeddings["raw"],
                true_labels,
                k_values,
                random_state,
            )
        )

        # Add UMAP tasks
        for dim in [2, 4, 8, 16, 32]:
            umap_key = f"umap_{dim}d"
            if umap_key in position_embeddings:
                tasks.append(
                    (
                        f"{position}_{umap_key}",
                        position_embeddings[umap_key],
                        true_labels,
                        k_values,
                        random_state,
                    )
                )
            else:
                print(f"  Warning: {umap_key} not found in {position} embeddings")

    # Process embedding tasks in parallel using concurrent.futures
    all_metrics = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [
            executor.submit(process_embedding_with_k_values, task) for task in tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc=f"Processing {fold_name} embeddings",
        ):
            try:
                result = future.result()
                all_metrics.extend(result)
            except Exception as e:
                print(f"Error processing embedding: {e}")

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


# Original function for backward compatibility
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

    # Store all metrics
    all_metrics = []

    # Process each embedding type
    for position in ["first", "half", "last"]:
        position_embeddings = embeddings[position]

        # Process raw embeddings
        print(f"  Processing raw {position} embeddings...")
        raw_results = process_embeddings(
            f"{position}_raw",
            position_embeddings["raw"],
            true_labels,
            k_values,
            random_state,
        )
        all_metrics.extend(raw_results)

        # Process UMAP projections
        for dim in [2, 4, 8, 16, 32]:
            umap_key = f"umap_{dim}d"
            print(f"  Processing {position} {umap_key}...")
            umap_results = process_embeddings(
                f"{position}_{umap_key}",
                position_embeddings[umap_key],
                true_labels,
                k_values,
                random_state,
            )
            all_metrics.extend(umap_results)

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

    # Implement nested parallelism with ThreadPoolExecutor for top level
    # and ProcessPoolExecutor for the inner level
    max_parallel_folds = min(args.max_parallel_folds, len(fold_files))
    fold_cpus = max(1, num_workers // max_parallel_folds)

    print(f"Using nested parallelism strategy:")
    print(f"  - Processing up to {max_parallel_folds} folds in parallel")
    print(f"  - Each fold will use up to {fold_cpus} CPUs")
    print(f"  - Total potential utilization: {max_parallel_folds * fold_cpus} CPUs")

    # Create task arguments for folds
    fold_tasks = [
        (fold_file, input_dir, output_dir, k_values, args.random_state, fold_cpus)
        for fold_file in fold_files
    ]

    # Use ThreadPoolExecutor for the outer level to avoid the daemon process issue
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_parallel_folds
    ) as executor:
        futures = [
            executor.submit(process_fold_with_parallelism, task) for task in fold_tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(fold_files),
            desc="Processing folds",
        ):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing fold: {e}")

    # Save a combined summary of all folds
    summaries = {fold_number: summary for fold_number, summary in results}
    summary_file = os.path.join(output_dir, "fold_summaries.pkl")
    with open(summary_file, "wb") as f:
        pickle.dump(summaries, f)

    # Print summary
    print("\nProcessing summary:")
    for fold_number, summary in sorted(results):
        print(
            f"  Fold {fold_number}: {summary['num_samples']} samples, {summary['unique_labels']} unique labels"
        )

    print(f"\nClustering metrics calculation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
