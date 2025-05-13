import argparse
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Control OpenBLAS thread usage to prevent thread explosion
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Numexpr threads

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
# Add OpenBLAS control parameters
parser.add_argument(
    "--blas_threads",
    type=int,
    default=1,
    help="Number of threads for BLAS operations per process. Default is 1 to prevent thread explosion.",
)
parser.add_argument(
    "--max_total_processes",
    type=int,
    default=128,
    help="Maximum total number of processes to use, to prevent OpenBLAS errors on large systems.",
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
    "--fold_workers",
    type=int,
    default=None,
    help="Number of worker processes for fold-level parallelism. If None, calculated automatically.",
)
parser.add_argument(
    "--embedding_workers",
    type=int,
    default=None,
    help="Number of worker processes for embedding-level parallelism per fold. If None, calculated automatically.",
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

# Set BLAS threading according to argument (done before any numpy imports)
if args.blas_threads > 1:
    print(f"Setting BLAS libraries to use {args.blas_threads} threads per process")
    os.environ["OMP_NUM_THREADS"] = str(args.blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.blas_threads)
else:
    print("Using single-threaded BLAS operations to maximize process parallelism")

# Setup directory paths
if args.ONLY_MAIN_LABEL:
    label_type = "main_labels_only"
else:
    label_type = "filtered_labels"

input_dir = os.path.join(args.base_dir, f"umap_data_{args.N}_{label_type}")
output_dir = os.path.join(args.base_dir, f"cluster_metrics_{args.N}_{label_type}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Calculate optimal number of worker processes
available_cpus = mp.cpu_count()

# If on a very large system, cap the total CPU usage more aggressively
if available_cpus > 64:
    print(f"Large system detected with {available_cpus} CPUs")
    effective_cpus = min(
        args.max_total_processes, int(available_cpus * args.cpu_fraction)
    )
    print(f"Capping effective CPUs to {effective_cpus} to prevent OpenBLAS issues")
else:
    effective_cpus = int(available_cpus * args.cpu_fraction)

# Set up hierarchical parallelism
if args.fold_workers is None:
    # For large systems, allocate CPUs intelligently between levels of parallelism
    if effective_cpus >= 32:
        fold_workers = min(
            4, effective_cpus // 8
        )  # Reduce from 8 to 4 for large systems
    else:
        fold_workers = max(1, effective_cpus // 4)  # At least one fold worker
else:
    fold_workers = args.fold_workers

# Calculate how many CPUs to use per fold
cpus_per_fold = max(1, effective_cpus // fold_workers)

# Set embedding workers if not specified
if args.embedding_workers is None:
    embedding_workers = max(1, cpus_per_fold // 2)  # Leave some CPUs for KMeans
else:
    embedding_workers = args.embedding_workers

print(f"Parallelization strategy:")
print(f"  Total available CPUs: {available_cpus}")
print(f"  Using {effective_cpus} CPUs ({args.cpu_fraction * 100:.0f}% of available)")
print(f"  Processing {fold_workers} folds in parallel")
print(f"  Using {embedding_workers} workers per fold for embedding parallelism")
print(f"  Effective CPUs per fold: ~{cpus_per_fold}")
print(f"  BLAS threads per process: {args.blas_threads}")

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


# Process a single embedding for all k values using sub-processes
def process_single_embedding(args):
    embedding_name, embeddings, true_labels, k_values, random_state = args
    results = []
    for k in k_values:
        metrics = calculate_metrics_for_k(embeddings, true_labels, k, random_state)
        metrics["embedding"] = embedding_name
        results.append(metrics)
    return results


# Function to process embeddings (UMAP or raw) in parallel
def process_embeddings_parallel(embedding_tasks, k_values, random_state, num_workers):
    # Prepare tasks
    tasks = [
        (name, emb, labels, k_values, random_state)
        for name, emb, labels in embedding_tasks
    ]

    # Process in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_embedding, tasks))

    # Flatten results
    all_results = []
    for result_batch in results:
        all_results.extend(result_batch)

    return all_results


# Function to process a single fold file
def process_fold(
    fold_file, input_dir, output_dir, k_values, random_state, embedding_workers
):
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

    # Prepare all embedding tasks
    embedding_tasks = []
    for position in ["first", "half", "last"]:
        position_embeddings = embeddings[position]

        # Add raw embeddings
        embedding_tasks.append(
            (f"{position}_raw", position_embeddings["raw"], true_labels)
        )

        # Add UMAP projections
        for dim in [2, 4, 8, 16, 32]:
            umap_key = f"umap_{dim}d"
            embedding_tasks.append(
                (f"{position}_{umap_key}", position_embeddings[umap_key], true_labels)
            )

    # Process all embeddings in parallel
    print(
        f"  Starting parallel processing of {len(embedding_tasks)} embedding types with {embedding_workers} workers"
    )
    all_metrics = process_embeddings_parallel(
        embedding_tasks, k_values, random_state, embedding_workers
    )

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

    # Set up the process pool and run the processing
    process_fold_partial = partial(
        process_fold,
        input_dir=input_dir,
        output_dir=output_dir,
        k_values=k_values,
        random_state=args.random_state,
        embedding_workers=embedding_workers,
    )

    # Process folds in parallel if fold_workers > 1, otherwise sequentially
    if fold_workers > 1:
        print(
            f"Using {fold_workers} worker processes for fold-level parallel processing"
        )
        with ProcessPoolExecutor(max_workers=fold_workers) as executor:
            # Using list() to ensure we wait for all tasks to complete
            results = list(executor.map(process_fold_partial, fold_files))
    else:
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
