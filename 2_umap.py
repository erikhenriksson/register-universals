import argparse
import multiprocessing as mp
import os
import pickle
import time
import warnings
from functools import partial

import numpy as np
import umap
from tqdm import tqdm

# Suppress specific UMAP warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Generate UMAP projections from embeddings."
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
    help="Random state for UMAP for reproducibility.",
)
parser.add_argument(
    "--n_neighbors", type=int, default=15, help="n_neighbors parameter for UMAP."
)
parser.add_argument(
    "--min_dist", type=float, default=0.1, help="min_dist parameter for UMAP."
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
args = parser.parse_args()

# Setup directory paths
if args.ONLY_MAIN_LABEL:
    label_type = "main_labels_only"
else:
    label_type = "filtered_labels"

input_dir = os.path.join(args.base_dir, f"sampled_data_{args.N}_{label_type}")
output_dir = os.path.join(args.base_dir, f"umap_data_{args.N}_{label_type}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Automatically determine number of worker processes
available_cpus = mp.cpu_count()
num_workers = max(1, int(available_cpus * args.cpu_fraction))
print(
    f"Detected {available_cpus} CPUs, using {num_workers} worker processes ({args.cpu_fraction * 100:.0f}%)"
)

# UMAP dimensions to generate
umap_dimensions = [2, 4, 8, 16, 32]


# Function to process a single fold file
def process_fold(fold_file, input_dir, output_dir, umap_dimensions, umap_params):
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

    # Process each embedding type
    results = {}

    # Process first token embeddings
    print(f"  Processing first token embeddings...")
    first_results = {"raw": embeds_first}
    for dim in umap_dimensions:
        print(f"    Generating {dim}D UMAP projection...")
        umap_model = umap.UMAP(n_components=dim, **umap_params)
        first_results[f"umap_{dim}d"] = umap_model.fit_transform(embeds_first)
    results["first"] = first_results

    # Process middle token embeddings
    print(f"  Processing middle token embeddings...")
    half_results = {"raw": embeds_half}
    for dim in umap_dimensions:
        print(f"    Generating {dim}D UMAP projection...")
        umap_model = umap.UMAP(n_components=dim, **umap_params)
        half_results[f"umap_{dim}d"] = umap_model.fit_transform(embeds_half)
    results["half"] = half_results

    # Process last token embeddings
    print(f"  Processing last token embeddings...")
    last_results = {"raw": embeds_last}
    for dim in umap_dimensions:
        print(f"    Generating {dim}D UMAP projection...")
        umap_model = umap.UMAP(n_components=dim, **umap_params)
        last_results[f"umap_{dim}d"] = umap_model.fit_transform(embeds_last)
    results["last"] = last_results

    # Create final results dictionary with metadata
    fold_results = {"metadata": metadata, "embeddings": results}

    # Save the results
    output_file = os.path.join(output_dir, f"umap_fold_{fold_number}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(fold_results, f)

    elapsed_time = time.time() - start_time
    print(
        f"  Completed {fold_name} in {elapsed_time:.2f} seconds. Saved to {output_file}"
    )

    return fold_number, len(fold_data)


# Main execution
def main():
    print("Starting UMAP projections generation...")
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
        umap_params=umap_params,
    )

    # Process folds in parallel if num_workers > 1, otherwise sequentially
    if num_workers > 1:
        print(f"Using {num_workers} worker processes for parallel processing")
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(
                pool.imap_unordered(process_fold_partial, fold_files),
                total=len(fold_files),
                desc="Processing folds",
            ):
                results.append(result)
    else:
        print("Processing sequentially (no multiprocessing)")
        results = []
        for fold_file in tqdm(fold_files, desc="Processing folds"):
            results.append(process_fold_partial(fold_file))

    # Print summary
    print("\nProcessing summary:")
    for fold_number, count in sorted(results):
        print(f"  Fold {fold_number}: {count} samples processed")

    print(f"\nUMAP projections generation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
