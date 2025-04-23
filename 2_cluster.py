import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import umap
from scipy import stats

# Parse command line arguments
parser = argparse.ArgumentParser(description="Analyze sampled data with clustering.")
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
args = parser.parse_args()

N = args.N
ONLY_MAIN_LABEL = args.ONLY_MAIN_LABEL
min_clusters = args.min_clusters
max_clusters = args.max_clusters

# Determine input directory based on parameters
if ONLY_MAIN_LABEL:
    input_dir = f"sampled_data_{N}_main_labels_only"
else:
    input_dir = f"sampled_data_{N}_filtered_labels"

print(f"Reading data from: {input_dir}")
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

print(f"Found {len(fold_files)} fold files.")

# Define the dimensionality reduction techniques and dimensions
reductions = {
    "raw": None,  # No reduction, use raw embeddings
    "pca": {"name": "PCA", "dims": [2, 4, 8, 16, 32]},
    "umap": {"name": "UMAP", "dims": [2, 4, 8, 16, 32]},
}

# Define embedding types to analyze
embedding_types = ["embed_first", "embed_half", "embed_last"]

# Storage for results
silhouette_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
davies_bouldin_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Process each fold
for fold_file in tqdm(fold_files, desc="Processing folds"):
    fold_path = os.path.join(input_dir, fold_file)

    # Load the fold data
    with open(fold_path, "rb") as f:
        fold_data = pickle.load(f)

    print(f"\nProcessing {fold_file} with {len(fold_data)} samples")

    # Group data by language to process separately
    lang_data = defaultdict(list)
    for item in fold_data:
        lang_data[item["lang"]].append(item)

    # Process each language separately
    for lang, items in lang_data.items():
        print(f"  Processing language: {lang} with {len(items)} samples")

        # Skip if too few samples
        if len(items) < max_clusters:
            print(
                f"    Skipping {lang}: Not enough samples ({len(items)}) for clustering"
            )
            continue

        # Process each embedding type
        for embed_type in embedding_types:
            # Extract embeddings
            embeddings = np.array([item[embed_type] for item in items])

            # Process with each reduction technique
            for reduction_key, reduction_config in reductions.items():
                if reduction_key == "raw":
                    # Use raw embeddings
                    reduced_data = embeddings
                    reduction_dims = [embeddings.shape[1]]  # Original dimensionality
                else:
                    reduction_dims = reduction_config["dims"]

                    for n_dims in reduction_dims:
                        if n_dims >= embeddings.shape[1]:
                            # Skip if requested dimensionality is higher than original
                            continue

                        # Apply dimensionality reduction
                        if reduction_key == "pca":
                            reducer = PCA(n_components=n_dims, random_state=42)
                            reduced_data = reducer.fit_transform(embeddings)
                        elif reduction_key == "umap":
                            reducer = umap.UMAP(
                                n_components=n_dims,
                                random_state=42,
                                n_neighbors=15,
                                min_dist=0.1,
                            )
                            reduced_data = reducer.fit_transform(embeddings)

                        reduction_name = (
                            f"{reduction_key}_{n_dims}"
                            if reduction_key != "raw"
                            else "raw"
                        )

                        # Evaluate clustering for different numbers of clusters
                        for n_clusters in range(min_clusters, max_clusters + 1):
                            kmeans = KMeans(
                                n_clusters=n_clusters, random_state=42, n_init=10
                            )
                            cluster_labels = kmeans.fit_predict(reduced_data)

                            # Calculate metrics if there are at least 2 clusters with data
                            if len(np.unique(cluster_labels)) >= 2:
                                sil_score = silhouette_score(
                                    reduced_data, cluster_labels
                                )
                                db_score = davies_bouldin_score(
                                    reduced_data, cluster_labels
                                )

                                # Store results
                                silhouette_results[lang][embed_type][
                                    reduction_name
                                ].append(sil_score)
                                davies_bouldin_results[lang][embed_type][
                                    reduction_name
                                ].append(db_score)
                            else:
                                print(
                                    f"    Warning: Only {len(np.unique(cluster_labels))} clusters found for {lang}, {embed_type}, {reduction_name}, {n_clusters} clusters"
                                )


# Calculate statistics across folds
def calculate_stats(results):
    stats = {}
    for lang in results:
        stats[lang] = {}
        for embed_type in results[lang]:
            stats[lang][embed_type] = {}
            for reduction_name in results[lang][embed_type]:
                # Convert to numpy array for easy stats calculation
                scores = np.array(results[lang][embed_type][reduction_name])

                # Calculate mean and standard error
                mean = np.mean(scores, axis=0)
                stderr = stats.sem(scores, axis=0)

                stats[lang][embed_type][reduction_name] = {
                    "mean": mean,
                    "stderr": stderr,
                }
    return stats


silhouette_stats = calculate_stats(silhouette_results)
davies_bouldin_stats = calculate_stats(davies_bouldin_results)


# Plot results
def plot_metric(stats, metric_name, output_prefix):
    # Get all languages and embedding types
    all_languages = list(stats.keys())

    for embed_type in embedding_types:
        plt.figure(figsize=(15, 8))

        # Different line styles for different reduction methods
        line_styles = {"raw": "-", "pca": "--", "umap": "-."}

        # Different colors for different dimensions
        color_map = plt.cm.get_cmap("viridis", 6)

        for lang_idx, lang in enumerate(all_languages):
            if embed_type not in stats[lang]:
                continue

            # Plot for each reduction method and dimension
            reduction_methods = defaultdict(list)

            for reduction_name in stats[lang][embed_type]:
                # Identify the base reduction method and dimension
                if reduction_name == "raw":
                    base_method = "raw"
                    dim = "raw"
                else:
                    base_method, dim = reduction_name.split("_")

                reduction_methods[(base_method, dim)].append(reduction_name)

            for method_idx, ((base_method, dim), _) in enumerate(
                sorted(reduction_methods.items())
            ):
                if base_method == "raw":
                    reduction_label = f"{lang} - Raw"
                    color = color_map(0)
                else:
                    dim_idx = reductions[base_method]["dims"].index(int(dim)) + 1
                    reduction_label = f"{lang} - {base_method.upper()} {dim}"
                    color = color_map(dim_idx)

                # Get the actual reduction name
                reduction_name = (
                    f"{base_method}_{dim}" if base_method != "raw" else "raw"
                )

                # Extract data
                mean = stats[lang][embed_type][reduction_name]["mean"]
                stderr = stats[lang][embed_type][reduction_name]["stderr"]

                # Create x-axis (number of clusters)
                x = np.arange(min_clusters, min_clusters + len(mean))

                # Plot mean and error band
                plt.plot(
                    x,
                    mean,
                    label=reduction_label,
                    linestyle=line_styles[base_method],
                    color=color,
                )
                plt.fill_between(
                    x, mean - stderr, mean + stderr, color=color, alpha=0.2
                )

        plt.xlabel("Number of Clusters")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} Score by Number of Clusters for {embed_type}")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="best")

        # Save plot
        output_file = f"{output_prefix}_{embed_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_file}")
        plt.close()


# Create output directory
output_dir = (
    f"clustering_results_{N}_{'main_labels' if ONLY_MAIN_LABEL else 'filtered_labels'}"
)
os.makedirs(output_dir, exist_ok=True)

# Plot silhouette scores
plot_metric(
    silhouette_stats, "Silhouette Score", os.path.join(output_dir, "silhouette")
)

# Plot Davies-Bouldin scores
plot_metric(
    davies_bouldin_stats,
    "Davies-Bouldin Score",
    os.path.join(output_dir, "davies_bouldin"),
)

print(f"\nClustering analysis complete. Results saved to {output_dir}/")
