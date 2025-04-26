import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
from scipy import stats
import multiprocessing as mp
import gc
import time

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Analyze sampled data with clustering (memory-efficient version)."
)
parser.add_argument(
    "--N", type=int, default=1000, help="Number of samples that was used."
)
parser.add_argument(
    "--ONLY_MAIN_LABEL", action="store_true", help="Whether only main labels were used."
)
parser.add_argument(
    "--min_clusters", type=int, default=2, help="Minimum number of clusters to try."
)
parser.add_argument(
    "--max_clusters",
    type=int,
    default=34,
    help="Maximum number of clusters to try (default: 4 languages × 8 labels + 2 = 34).",
)
parser.add_argument(
    "--n_jobs",
    type=int,
    default=None,
    help="Number of parallel jobs to run. Default: 50% of available CPUs.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=50,
    help="Number of clustering tasks to process in one batch to manage memory usage.",
)
parser.add_argument(
    "--embedding_type",
    type=str,
    default=None,
    choices=["embed_first", "embed_half", "embed_last"],
    help="Process only a specific embedding type. If not specified, processes all types one by one.",
)
parser.add_argument(
    "--use_saved_results",
    action="store_true",
    help="Use previously saved intermediate results if available.",
)
parser.add_argument(
    "--resume_from",
    type=str,
    default=None,
    help="Resume from a specific fold file, skipping earlier folds.",
)
args = parser.parse_args()

N = args.N
ONLY_MAIN_LABEL = args.ONLY_MAIN_LABEL
min_clusters = args.min_clusters
max_clusters = args.max_clusters

# Use fewer CPUs by default to leave more memory per process
if args.n_jobs is None:
    available_cpus = mp.cpu_count()
    n_jobs = max(1, int(available_cpus * 0.5))  # Use 50% of CPUs by default
else:
    n_jobs = args.n_jobs

print(f"Using {n_jobs} worker processes for parallel processing")

# Determine input directory based on parameters
if ONLY_MAIN_LABEL:
    input_dir = f"sampled_data_{N}_main_labels_only"
else:
    input_dir = f"sampled_data_{N}_filtered_labels"

# Create output directory
output_dir = (
    f"clustering_results_{N}_{'main_labels' if ONLY_MAIN_LABEL else 'filtered_labels'}"
)
os.makedirs(output_dir, exist_ok=True)

print(f"Reading data from: {input_dir}")
print(f"Results will be saved to: {output_dir}")

if not os.path.exists(input_dir):
    print(
        f"Error: Directory {input_dir} does not exist. Please run the sampling script first."
    )
    exit(1)

# Get list of fold pickle files
fold_files = [
    f for f in os.listdir(input_dir) if f.startswith("fold_") and f.endswith(".pkl")
]
fold_files.sort()

# If resuming from a specific fold, filter the fold files
if args.resume_from:
    if args.resume_from in fold_files:
        resume_index = fold_files.index(args.resume_from)
        skipped_folds = fold_files[:resume_index]
        fold_files = fold_files[resume_index:]
        print(
            f"Resuming from {args.resume_from}. Skipping {len(skipped_folds)} earlier folds."
        )
    else:
        print(
            f"Warning: Specified resume fold {args.resume_from} not found. Processing all folds."
        )

print(f"Found {len(fold_files)} fold files to process.")

# Define embedding types to analyze
if args.embedding_type:
    embedding_types = [args.embedding_type]
    print(f"Processing only embedding type: {args.embedding_type}")
else:
    embedding_types = ["embed_first", "embed_half", "embed_last"]
    print(f"Processing all embedding types sequentially: {embedding_types}")

# Define the dimensionality reduction techniques and dimensions
reductions = {
    "raw": None,  # No reduction, use raw embeddings
    "umap": {"name": "UMAP", "dims": [2, 4, 8, 16, 32]},
}

# Storage for results
silhouette_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
davies_bouldin_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


# Function to process a single clustering task
def process_clustering_task(task_data):
    lang, embed_type, reduction_name, n_clusters, reduced_data = task_data

    try:
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced_data)

        # Calculate metrics if there are at least 2 clusters with data
        if len(np.unique(cluster_labels)) >= 2:
            sil_score = silhouette_score(reduced_data, cluster_labels)
            db_score = davies_bouldin_score(reduced_data, cluster_labels)
            return (
                lang,
                embed_type,
                reduction_name,
                n_clusters,
                True,
                sil_score,
                db_score,
            )
        else:
            print(
                f"    Warning: Only {len(np.unique(cluster_labels))} clusters found for {lang}, {embed_type}, {reduction_name}, {n_clusters} clusters"
            )
            return (lang, embed_type, reduction_name, n_clusters, False, None, None)
    except Exception as e:
        print(
            f"    Error processing {lang}, {embed_type}, {reduction_name}, {n_clusters} clusters: {str(e)}"
        )
        return (lang, embed_type, reduction_name, n_clusters, False, None, None)


# Process each fold
for fold_file in tqdm(fold_files, desc="Processing folds"):
    fold_path = os.path.join(input_dir, fold_file)
    fold_results_file = os.path.join(output_dir, f"fold_results_{fold_file}")

    # Skip if results already exist and we're using saved results
    if os.path.exists(fold_results_file) and args.use_saved_results:
        print(f"Loading existing results for {fold_file}")
        with open(fold_results_file, "rb") as f:
            fold_results = pickle.load(f)

        # Update the global results
        for lang, embed_results in fold_results["silhouette"].items():
            for embed_type, reduction_results in embed_results.items():
                for reduction_name, scores in reduction_results.items():
                    silhouette_results[lang][embed_type][reduction_name] = scores

        for lang, embed_results in fold_results["davies_bouldin"].items():
            for embed_type, reduction_results in embed_results.items():
                for reduction_name, scores in reduction_results.items():
                    davies_bouldin_results[lang][embed_type][reduction_name] = scores

        continue

    # Load the fold data
    print(f"\nProcessing {fold_file}")
    with open(fold_path, "rb") as f:
        fold_data = pickle.load(f)

    print(f"Loaded {fold_file} with {len(fold_data)} samples")

    # Group data by language
    lang_data = defaultdict(list)
    for item in fold_data:
        lang_data[item["lang"]].append(item)

    # Clear fold_data to free memory
    del fold_data
    gc.collect()

    # Initialize fold-specific results
    fold_silhouette = defaultdict(lambda: defaultdict(dict))
    fold_davies_bouldin = defaultdict(lambda: defaultdict(dict))

    # Process each embedding type separately to save memory
    for embed_type in embedding_types:
        print(f"\n  Processing embedding type: {embed_type}")

        # Process each language separately
        for lang, items in lang_data.items():
            print(f"    Processing language: {lang} with {len(items)} samples")

            # Skip if too few samples
            if len(items) < max_clusters:
                print(
                    f"      Skipping {lang}: Not enough samples ({len(items)}) for clustering"
                )
                continue

            # Extract only the needed embeddings to save memory
            embeddings = np.array([item[embed_type] for item in items])

            # Prepare for each reduction technique
            for reduction_key, reduction_config in reductions.items():
                if reduction_key == "raw":
                    # Use raw embeddings
                    reduced_data = embeddings
                    reduction_name = "raw"

                    # Prepare clustering tasks
                    tasks = []
                    for n_clusters in range(min_clusters, max_clusters + 1):
                        task = (
                            lang,
                            embed_type,
                            reduction_name,
                            n_clusters,
                            reduced_data,
                        )
                        tasks.append(task)

                    # Process in batches
                    print(
                        f"      Processing {len(tasks)} raw data clustering tasks in batches"
                    )
                    results = []

                    for i in range(0, len(tasks), args.batch_size):
                        batch = tasks[i : i + args.batch_size]
                        with mp.Pool(processes=n_jobs) as pool:
                            batch_results = list(
                                tqdm(
                                    pool.imap(process_clustering_task, batch),
                                    total=len(batch),
                                    desc=f"      Batch {i//args.batch_size + 1}/{(len(tasks) + args.batch_size - 1)//args.batch_size}",
                                )
                            )
                            results.extend(batch_results)

                        # Force garbage collection between batches
                        gc.collect()
                        time.sleep(1)  # Small delay to ensure memory is freed
                else:
                    reduction_dims = reduction_config["dims"]

                    for n_dims in reduction_dims:
                        if n_dims >= embeddings.shape[1]:
                            # Skip if requested dimensionality is higher than original
                            continue

                        # Apply dimensionality reduction
                        if reduction_key == "umap":
                            print(
                                f"      Applying UMAP reduction to {n_dims} dimensions"
                            )
                            reducer = umap.UMAP(
                                n_components=n_dims,
                                random_state=42,
                                n_neighbors=15,
                                min_dist=0.1,
                            )
                            reduced_data = reducer.fit_transform(embeddings)

                        reduction_name = f"{reduction_key}_{n_dims}"

                        # Prepare clustering tasks
                        tasks = []
                        for n_clusters in range(min_clusters, max_clusters + 1):
                            task = (
                                lang,
                                embed_type,
                                reduction_name,
                                n_clusters,
                                reduced_data,
                            )
                            tasks.append(task)

                        # Process in batches
                        print(
                            f"      Processing {len(tasks)} {reduction_name} clustering tasks in batches"
                        )
                        results = []

                        for i in range(0, len(tasks), args.batch_size):
                            batch = tasks[i : i + args.batch_size]
                            with mp.Pool(processes=n_jobs) as pool:
                                batch_results = list(
                                    tqdm(
                                        pool.imap(process_clustering_task, batch),
                                        total=len(batch),
                                        desc=f"      Batch {i//args.batch_size + 1}/{(len(tasks) + args.batch_size - 1)//args.batch_size}",
                                    )
                                )
                                results.extend(batch_results)

                            # Force garbage collection between batches
                            gc.collect()
                            time.sleep(1)  # Small delay to ensure memory is freed

                # Process results for this reduction method
                for (
                    lang,
                    et,
                    red_name,
                    n_clusters,
                    success,
                    sil_score,
                    db_score,
                ) in results:
                    if success:
                        # Initialize if needed
                        if red_name not in fold_silhouette[lang][et]:
                            fold_silhouette[lang][et][red_name] = [None] * (
                                max_clusters - min_clusters + 1
                            )
                            fold_davies_bouldin[lang][et][red_name] = [None] * (
                                max_clusters - min_clusters + 1
                            )

                        # Store scores at the correct index
                        idx = n_clusters - min_clusters
                        fold_silhouette[lang][et][red_name][idx] = sil_score
                        fold_davies_bouldin[lang][et][red_name][idx] = db_score

    # Save fold results
    fold_results = {
        "silhouette": fold_silhouette,
        "davies_bouldin": fold_davies_bouldin,
    }

    with open(fold_results_file, "wb") as f:
        pickle.dump(fold_results, f)
    print(f"Saved results for {fold_file} to {fold_results_file}")

    # Update the global results
    for lang, embed_results in fold_silhouette.items():
        for embed_type, reduction_results in embed_results.items():
            for reduction_name, scores in reduction_results.items():
                silhouette_results[lang][embed_type][reduction_name].append(scores)

    for lang, embed_results in fold_davies_bouldin.items():
        for embed_type, reduction_results in embed_results.items():
            for reduction_name, scores in reduction_results.items():
                davies_bouldin_results[lang][embed_type][reduction_name].append(scores)


# Calculate statistics across folds
def calculate_stats(results):
    stats = {}
    for lang in results:
        stats[lang] = {}
        for embed_type in results[lang]:
            stats[lang][embed_type] = {}
            for reduction_name in results[lang][embed_type]:
                # Filter out None values for each cluster count
                processed_scores = []
                for fold_scores in results[lang][embed_type][reduction_name]:
                    if fold_scores is not None:
                        processed_scores.append(
                            [
                                score if score is not None else np.nan
                                for score in fold_scores
                            ]
                        )

                if not processed_scores:
                    continue

                # Convert to numpy array for easy stats calculation
                scores_array = np.array(processed_scores)

                # Calculate mean and standard error, ignoring NaNs
                mean = np.nanmean(scores_array, axis=0)
                stderr = stats.sem(scores_array, axis=0, nan_policy="omit")

                stats[lang][embed_type][reduction_name] = {
                    "mean": mean,
                    "stderr": stderr,
                }
    return stats


silhouette_stats = calculate_stats(silhouette_results)
davies_bouldin_stats = calculate_stats(davies_bouldin_results)

# Save final results
results_data = {
    "silhouette_results": silhouette_results,
    "davies_bouldin_results": davies_bouldin_results,
    "silhouette_stats": silhouette_stats,
    "davies_bouldin_stats": davies_bouldin_stats,
    "parameters": {
        "N": N,
        "ONLY_MAIN_LABEL": ONLY_MAIN_LABEL,
        "min_clusters": min_clusters,
        "max_clusters": max_clusters,
        "embedding_types": embedding_types,
        "languages": list(silhouette_results.keys()),
    },
}

results_file = os.path.join(output_dir, "clustering_results.pkl")
with open(results_file, "wb") as f:
    pickle.dump(results_data, f)
print(f"Saved final results to {results_file}")
print(
    "\nClustering analysis complete. Use plot_clustering_results.py to visualize the results."
)
