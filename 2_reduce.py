import argparse
import multiprocessing as mp
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Control OpenBLAS thread usage to prevent thread explosion
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "1"  # MKL threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Numexpr threads

import numpy as np
import umap
from sklearn.decomposition import PCA
from tqdm import tqdm

# Suppress specific UMAP warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Generate UMAP and PCA projections from embeddings."
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
    help="Random state for UMAP and PCA for reproducibility.",
)
parser.add_argument(
    "--n_neighbors", type=int, default=50, help="n_neighbors parameter for UMAP."
)
parser.add_argument(
    "--min_dist", type=float, default=0.0, help="min_dist parameter for UMAP."
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="./",
    help="Base directory containing sampled data directories.",
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
    "--blas_threads",
    type=int,
    default=1,
    help="Number of threads for BLAS operations per process. Default is 1 to prevent thread explosion.",
)
parser.add_argument(
    "--max_total_processes",
    type=int,
    default=64,
    help="Maximum total number of processes to use, to prevent OpenBLAS errors on large systems.",
)
args = parser.parse_args()

# Set BLAS threading according to argument
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

input_dir = os.path.join(args.base_dir, f"sampled_data_{args.N}_{label_type}")
output_dir = os.path.join(
    args.base_dir, f"dimensionality_reduction_{args.N}_{label_type}"
)

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
    embedding_workers = max(1, cpus_per_fold // 2)  # Leave some CPUs for UMAP/PCA
else:
    embedding_workers = args.embedding_workers

print(f"Parallelization strategy:")
print(f"  Total available CPUs: {available_cpus}")
print(f"  Using {effective_cpus} CPUs ({args.cpu_fraction * 100:.0f}% of available)")
print(f"  Processing {fold_workers} folds in parallel")
print(f"  Using {embedding_workers} workers per fold for embedding parallelism")
print(f"  Effective CPUs per fold: ~{cpus_per_fold}")
print(f"  BLAS threads per process: {args.blas_threads}")

# Dimensionality reduction configurations
umap_dimensions = [2, 4, 8, 16, 32]
pca_dimensions = [2, 4, 8, 16]
pca_umap_dimension = 2  # UMAP dimension to apply on PCA results


# Function to process a single PCA embedding-dimension combination
def process_pca_embedding(args):
    embedding, dim, random_state = args
    try:
        pca_model = PCA(n_components=dim, random_state=random_state)
        result = pca_model.fit_transform(embedding)
        explained_variance = pca_model.explained_variance_ratio_.sum()
        return dim, result, explained_variance
    except Exception as e:
        print(f"Error processing {dim}D PCA: {str(e)}")
        return dim, None, 0.0


# Function to process UMAP on a PCA-reduced embedding
def process_pca_umap_embedding(args):
    pca_embedding, dim, umap_params = args
    try:
        umap_model = umap.UMAP(
            n_components=2, **umap_params
        )  # Always use 2D for PCA+UMAP
        result = umap_model.fit_transform(pca_embedding)
        return dim, result
    except Exception as e:
        print(f"Error processing UMAP on {dim}D PCA: {str(e)}")
        return dim, None


# Function to process a single UMAP embedding-dimension combination
def process_umap_embedding(args):
    embedding, dim, umap_params = args
    try:
        umap_model = umap.UMAP(n_components=dim, **umap_params)
        result = umap_model.fit_transform(embedding)
        return dim, result
    except Exception as e:
        print(f"Error processing {dim}D UMAP: {str(e)}")
        return dim, None


# Process a single position's embeddings with parallel PCA, UMAP, and PCA+UMAP
def process_position_embeddings(
    position_name,
    embeddings,
    umap_dimensions,
    pca_dimensions,
    umap_params,
    random_state,
    num_workers,
):
    print(
        f"  Processing {position_name} token embeddings with {num_workers} workers..."
    )

    results = {"raw": embeddings}

    # 1. Prepare PCA tasks for parallel execution
    pca_tasks = [(embeddings, dim, random_state) for dim in pca_dimensions]

    pca_results = {}

    # Run PCA in parallel for different dimensions
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for dim, result, explained_variance in executor.map(
            process_pca_embedding, pca_tasks
        ):
            if result is not None:
                pca_results[dim] = result
                results[f"pca_{dim}d"] = result
                print(
                    f"    Completed {dim}D PCA projection for {position_name} tokens "
                    f"(explained variance: {explained_variance:.2%})"
                )

    # Add this section - import sys at the top of the script if not already there
    import sys

    print(
        f"PCA for {position_name} tokens completed. Exiting early to prevent further processing."
    )
    sys.exit(0)  # Exit with success code


# Function to process a single fold file
def process_fold(
    fold_file,
    input_dir,
    output_dir,
    umap_dimensions,
    pca_dimensions,
    umap_params,
    random_state,
    embedding_workers,
):
    start_time = time.time()
    fold_name = os.path.basename(fold_file)
    fold_number = fold_name.split("_")[1].split(".")[0]

    print(f"Processing {fold_name}...")

    # Load the fold data
    fold_path = os.path.join(input_dir, fold_file)
    with open(fold_path, "rb") as f:
        fold_data = pickle.load(f)

    print(f"  Loaded {len(fold_data)} samples from {fold_name}")

    # Extract embeddings
    embeds_first = np.array([item["embed_first"] for item in fold_data])
    embeds_half = np.array([item["embed_half"] for item in fold_data])
    embeds_last = np.array([item["embed_last"] for item in fold_data])

    # Extract metadata for later use
    metadata = [
        {"lang": item["lang"], "text": item["text"], "preds_0.4": item["preds_0.4"]}
        for item in fold_data
    ]

    # Process each embedding type in parallel
    results = {}

    # Process all three positions (first, half, last) sequentially
    # Each position will process its different PCA/UMAP dimensions in parallel
    results["first"] = process_position_embeddings(
        "first",
        embeds_first,
        umap_dimensions,
        pca_dimensions,
        umap_params,
        random_state,
        embedding_workers,
    )

    results["half"] = process_position_embeddings(
        "half",
        embeds_half,
        umap_dimensions,
        pca_dimensions,
        umap_params,
        random_state,
        embedding_workers,
    )

    results["last"] = process_position_embeddings(
        "last",
        embeds_last,
        umap_dimensions,
        pca_dimensions,
        umap_params,
        random_state,
        embedding_workers,
    )

    # Create final results dictionary with metadata
    fold_results = {"metadata": metadata, "embeddings": results}

    # Save the results
    output_file = os.path.join(output_dir, f"dim_reduction_fold_{fold_number}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(fold_results, f)

    elapsed_time = time.time() - start_time
    print(
        f"  Completed {fold_name} in {elapsed_time:.2f} seconds. Saved to {output_file}"
    )

    return fold_number, len(fold_data)


# Main execution
def main():
    print("Starting dimensionality reduction projections generation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # List fold files
    fold_files = [
        f for f in os.listdir(input_dir) if f.startswith("fold_") and f.endswith(".pkl")
    ]
    fold_files.sort()  # Sort to process in order

    if not fold_files:
        print(f"Error: No fold files found in {input_dir}")
        return

    print(f"Found {len(fold_files)} fold files to process")

    # UMAP parameters
    umap_params = {
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "random_state": args.random_state,
        "metric": "cosine",  # Using cosine distance which is common for embeddings
    }

    # Set up the process pool and run the processing
    process_fold_partial = partial(
        process_fold,
        input_dir=input_dir,
        output_dir=output_dir,
        umap_dimensions=umap_dimensions,
        pca_dimensions=pca_dimensions,
        umap_params=umap_params,
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

    # Print summary
    print("\nProcessing summary:")
    for fold_number, count in sorted(results, key=lambda x: x[0]):
        print(f"  Fold {fold_number}: {count} samples processed")

    print("\nDimensionality reduction summary:")
    print("  PCA dimensions: ", pca_dimensions)
    print("  UMAP dimensions: ", umap_dimensions)
    print("  PCA+UMAP: PCA dimensions", pca_dimensions, "followed by 2D UMAP")

    print(f"\nAll dimensionality reductions complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
