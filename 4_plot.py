import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate plots from clustering metrics.")
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
    "--base_dir",
    type=str,
    default="./",
    help="Base directory containing cluster metrics directories.",
)
parser.add_argument(
    "--metrics",
    nargs="+",
    default=[
        "silhouette",
        "ari",
        "nmi",
        "v_measure",
        "homogeneity",
        "completeness",
        "davies_bouldin",
        "calinski_harabasz",
    ],
    help="Metrics to plot (default: silhouette). Options: silhouette, ari, nmi, "
    + "v_measure, homogeneity, completeness, davies_bouldin, calinski_harabasz",
)
parser.add_argument(
    "--embedding_positions",
    nargs="+",
    default=["first", "half", "last"],
    help="Embedding positions to plot (default: first half last).",
)
parser.add_argument(
    "--embedding_types",
    nargs="+",
    default=["raw", "umap_2d", "umap_4d", "umap_8d", "umap_16d", "umap_32d"],
    help="Embedding types to plot (default: raw and all UMAP dimensions).",
)
parser.add_argument(
    "--output_format",
    type=str,
    default="png",
    choices=["png", "pdf", "svg"],
    help="Output file format (default: png).",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=300,
    help="DPI for output images (default: 300).",
)
parser.add_argument(
    "--show_plots",
    action="store_true",
    help="Show plots during execution.",
)
parser.add_argument(
    "--plot_umap",
    action="store_true",
    default=True,
    help="Generate UMAP 2D scatter plots colored by register (default: True).",
)
parser.add_argument(
    "--umap_dir",
    type=str,
    default=None,
    help="Directory containing UMAP data. If None, will use base_dir/umap_data_N_label_type.",
)
# NEW PARAMETER: Add cluster numbers to visualize in k-means plots
parser.add_argument(
    "--kmeans_clusters",
    nargs="+",
    type=int,
    default=[4, 8, 25],
    help="K values to use for k-means cluster visualization (default: [4, 8, 25]).",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug printing for diagnosing issues.",
)
args = parser.parse_args()

# Setup directory paths
if args.ONLY_MAIN_LABEL:
    label_type = "main_labels_only"
else:
    label_type = "filtered_labels"

input_dir = os.path.join(args.base_dir, f"cluster_metrics_{args.N}_{label_type}")
output_dir = os.path.join(args.base_dir, f"plots_{args.N}_{label_type}")

# Set UMAP directory
if args.umap_dir is None:
    umap_dir = os.path.join(args.base_dir, f"umap_data_{args.N}_{label_type}")
else:
    umap_dir = args.umap_dir

# Create output directory
os.makedirs(output_dir, exist_ok=True)


def load_metrics_data():
    """Load metrics data from all fold files and organize it."""
    print(f"Loading metrics data from {input_dir}...")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return None

    # List fold files
    fold_files = [
        f
        for f in os.listdir(input_dir)
        if f.startswith("metrics_fold_") and f.endswith(".pkl")
    ]
    fold_files.sort()

    if not fold_files:
        print(f"Error: No metrics fold files found in {input_dir}")
        return None

    print(f"Found {len(fold_files)} fold files to process")

    # Dictionary to store all metrics data
    all_data = {}

    # Load and process each fold file
    for fold_file in tqdm(fold_files, desc="Loading fold files"):
        fold_path = os.path.join(input_dir, fold_file)
        with open(fold_path, "rb") as f:
            fold_metrics = pickle.load(f)

        fold_number = fold_file.split("_")[2].split(".")[0]

        # Process each metric
        for metric_data in fold_metrics:
            embedding = metric_data["embedding"]
            k = metric_data["k"]

            # Initialize embedding in all_data if not present
            if embedding not in all_data:
                all_data[embedding] = {}

            # Initialize k in embedding if not present
            if k not in all_data[embedding]:
                all_data[embedding][k] = {}

            # Store each metric for this embedding, k, and fold
            for metric in args.metrics:
                if metric not in all_data[embedding][k]:
                    all_data[embedding][k][metric] = {}

                all_data[embedding][k][metric][fold_number] = metric_data.get(metric)

            # Store k-means cluster labels if present
            if "cluster_labels" in metric_data:
                if "cluster_labels" not in all_data[embedding][k]:
                    all_data[embedding][k]["cluster_labels"] = {}

                all_data[embedding][k]["cluster_labels"][fold_number] = metric_data.get(
                    "cluster_labels"
                )

    return all_data


def process_metrics_data(all_data):
    """
    Process the loaded metrics data to calculate averages and ranges.
    Returns a dictionary with calculated statistics for each metric, embedding, and k.
    """
    print("Processing metrics data...")

    # Dictionary to store processed data
    processed_data = {}

    for embedding in all_data:
        processed_data[embedding] = {}

        # Get all k values and sort them
        k_values = sorted(all_data[embedding].keys())

        for metric in args.metrics:
            processed_data[embedding][metric] = {
                "k_values": k_values,
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
                "fold_data": {},  # Store data for each fold for detailed plotting
            }

            for k in k_values:
                # Collect all non-None values for this metric across folds
                metric_values = [
                    v for v in all_data[embedding][k][metric].values() if v is not None
                ]

                # Store data for each fold
                for fold, value in all_data[embedding][k][metric].items():
                    if fold not in processed_data[embedding][metric]["fold_data"]:
                        processed_data[embedding][metric]["fold_data"][fold] = []
                    processed_data[embedding][metric]["fold_data"][fold].append(value)

                # Calculate statistics if we have data
                if metric_values:
                    processed_data[embedding][metric]["mean"].append(
                        np.mean(metric_values)
                    )
                    processed_data[embedding][metric]["std"].append(
                        np.std(metric_values)
                    )
                    processed_data[embedding][metric]["min"].append(
                        np.min(metric_values)
                    )
                    processed_data[embedding][metric]["max"].append(
                        np.max(metric_values)
                    )
                else:
                    # No valid data for this k
                    processed_data[embedding][metric]["mean"].append(None)
                    processed_data[embedding][metric]["std"].append(None)
                    processed_data[embedding][metric]["min"].append(None)
                    processed_data[embedding][metric]["max"].append(None)

    return processed_data


def plot_metric(metric, processed_data):
    """Create plots for a specific metric across different embeddings."""
    print(f"Plotting {metric} scores...")

    # Define a color palette for different embedding types
    palette = sns.color_palette("husl", len(args.embedding_types))
    embedding_colors = {et: palette[i] for i, et in enumerate(args.embedding_types)}

    # Plot for each embedding position (first, half, last)
    for position in args.embedding_positions:
        plt.figure(figsize=(12, 8))

        # Plot each embedding type (raw, umap_2d, etc.)
        for i, embed_type in enumerate(args.embedding_types):
            embed_name = f"{position}_{embed_type}"

            # Skip if this embedding is not in the data
            if embed_name not in processed_data:
                print(f"Warning: {embed_name} not found in data, skipping.")
                continue

            # Get data for this embedding
            embed_data = processed_data[embed_name][metric]
            k_values = embed_data["k_values"]
            mean_values = embed_data["mean"]

            # Filter out None values
            valid_indices = [i for i, v in enumerate(mean_values) if v is not None]
            if not valid_indices:
                print(f"Warning: No valid data for {embed_name}, skipping.")
                continue

            valid_k = [k_values[i] for i in valid_indices]
            valid_mean = [mean_values[i] for i in valid_indices]

            # For standard deviation shading
            if args.embedding_types.index(embed_type) == 0:  # For first type, use std
                lower_bound = [
                    embed_data["mean"][i] - embed_data["std"][i]
                    if embed_data["mean"][i] is not None
                    else None
                    for i in range(len(k_values))
                ]
                upper_bound = [
                    embed_data["mean"][i] + embed_data["std"][i]
                    if embed_data["mean"][i] is not None
                    else None
                    for i in range(len(k_values))
                ]
            else:  # For others, use min/max for clearer distinction
                lower_bound = embed_data["min"]
                upper_bound = embed_data["max"]

            valid_lower = [lower_bound[i] for i in valid_indices]
            valid_upper = [upper_bound[i] for i in valid_indices]

            # Plot mean line
            plt.plot(
                valid_k,
                valid_mean,
                label=embed_type,
                color=embedding_colors[embed_type],
                linewidth=2,
            )

            # Plot shaded area for range
            plt.fill_between(
                valid_k,
                valid_lower,
                valid_upper,
                alpha=0.1,
                color=embedding_colors[embed_type],
            )

        # Customize plot
        if metric == "silhouette":
            plt.title(
                f"Silhouette Score vs. Number of Clusters ({position} embeddings)"
            )
            plt.ylabel("Silhouette Score")
            plt.ylim(-0.1, 1.0)  # Silhouette score ranges from -1 to 1
        elif metric == "ari":
            plt.title(
                f"Adjusted Rand Index vs. Number of Clusters ({position} embeddings)"
            )
            plt.ylabel("Adjusted Rand Index")
        elif metric == "nmi":
            plt.title(
                f"Normalized Mutual Information vs. Number of Clusters ({position} embeddings)"
            )
            plt.ylabel("NMI Score")
        elif metric == "davies_bouldin":
            plt.title(
                f"Davies-Bouldin Index vs. Number of Clusters ({position} embeddings)"
            )
            plt.ylabel("Davies-Bouldin Index (lower is better)")
        elif metric == "calinski_harabasz":
            plt.title(
                f"Calinski-Harabasz Index vs. Number of Clusters ({position} embeddings)"
            )
            plt.ylabel("Calinski-Harabasz Index (higher is better)")
        else:
            plt.title(
                f"{metric.capitalize()} Score vs. Number of Clusters ({position} embeddings)"
            )
            plt.ylabel(f"{metric.capitalize()} Score")

        plt.xlabel("Number of Clusters (k)")
        plt.legend(title="Embedding Type", loc="best")
        plt.grid(True)

        # Save plot
        output_file = os.path.join(
            output_dir, f"{metric}_{position}.{args.output_format}"
        )
        plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved plot to {output_file}")

        # Show plot if requested
        if args.show_plots:
            plt.show()
        else:
            plt.close()


def get_majority_labels(metadata):
    """Get the majority label for each sample."""
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


def plot_umap_scatter(umap_dir, all_data=None):
    """Create scatter plots of 2D UMAP projections colored by register and k-means clusters."""
    print(f"Plotting UMAP 2D projections from {umap_dir}...")

    # Check if UMAP directory exists
    if not os.path.exists(umap_dir):
        print(f"Error: UMAP directory {umap_dir} does not exist.")
        return

    # List fold files
    fold_files = [
        f
        for f in os.listdir(umap_dir)
        if f.startswith("umap_fold_") and f.endswith(".pkl")
    ]
    fold_files.sort()

    if not fold_files:
        print(f"Error: No UMAP fold files found in {umap_dir}")
        return

    print(f"Found {len(fold_files)} UMAP fold files to process")

    # Process each fold
    for fold_file in tqdm(fold_files, desc="Creating UMAP plots"):
        fold_path = os.path.join(umap_dir, fold_file)
        fold_number = fold_file.split("_")[2].split(".")[0]

        try:
            # Load the fold data
            with open(fold_path, "rb") as f:
                fold_data = pickle.load(f)

            metadata = fold_data.get("metadata", [])
            embeddings = fold_data.get("embeddings", {})

            if not metadata or not embeddings:
                print(f"Warning: Missing metadata or embeddings in {fold_file}")
                continue

            # Get labels for coloring points
            labels = get_majority_labels(metadata)
            unique_labels = np.unique(labels)

            # Create a colormap for the labels
            num_labels = len(unique_labels)
            if num_labels <= 10:
                # Use a categorical palette for few labels
                cmap = plt.cm.get_cmap("tab10", num_labels)
            else:
                # Use a continuous colormap for many labels
                cmap = plt.cm.get_cmap("hsv", num_labels)

            # Create a dictionary mapping labels to colors
            label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

            # Create scatter plots for each position (first, half, last)
            for position in args.embedding_positions:
                if position not in embeddings:
                    print(f"Warning: Position {position} not found in embeddings")
                    continue

                position_embeddings = embeddings[position]

                # Get 2D UMAP projections
                if "umap_2d" not in position_embeddings:
                    print(f"Warning: 2D UMAP not found for {position} embeddings")
                    continue

                umap_2d = position_embeddings["umap_2d"]

                # Create figure
                plt.figure(figsize=(12, 10))

                # Plot each label with its own color
                for label in unique_labels:
                    mask = labels == label
                    plt.scatter(
                        umap_2d[mask, 0],
                        umap_2d[mask, 1],
                        c=[label_colors[label]],
                        label=label,
                        alpha=0.7,
                        s=5,
                        edgecolors="none",
                    )

                # Add labels and legend
                plt.title(
                    f"2D UMAP Projection ({position} embeddings), Fold {fold_number}"
                )
                plt.xlabel("UMAP Dimension 1")
                plt.ylabel("UMAP Dimension 2")

                # Add legend with reasonable size and position - FIXED
                if num_labels <= 20:
                    # Full legend for manageable number of labels
                    legend = plt.legend(
                        title="Register",
                        loc="best",
                        bbox_to_anchor=(1.05, 1),
                        fontsize=10,
                        markerscale=10,  # Scale marker size in legend
                    )
                else:
                    # Create a more compact legend for many labels
                    legend = plt.legend(
                        title="Register",
                        loc="center left",
                        bbox_to_anchor=(1.05, 0.5),
                        fontsize=8,
                        ncol=2,
                        markerscale=8,  # Scale marker size in legend
                    )

                plt.tight_layout()

                # Save plot
                output_file = os.path.join(
                    output_dir,
                    f"umap_2d_{position}_fold_{fold_number}.{args.output_format}",
                )
                plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved plot to {output_file}")

                # Show plot if requested
                if args.show_plots:
                    plt.show()
                else:
                    plt.close()

            # Create aggregate plot of all languages for each position
            for position in args.embedding_positions:
                if position not in embeddings:
                    continue

                position_embeddings = embeddings[position]

                if "umap_2d" not in position_embeddings:
                    continue

                umap_2d = position_embeddings["umap_2d"]

                # Get language info for each point
                languages = np.array([item["lang"] for item in metadata])
                unique_languages = np.unique(languages)

                # Create a colormap for languages
                num_languages = len(unique_languages)
                cmap_lang = plt.cm.get_cmap("Set1", max(num_languages, 9))

                # Create figure
                plt.figure(figsize=(12, 10))

                # Plot each language with its own color
                for i, lang in enumerate(unique_languages):
                    mask = languages == lang
                    plt.scatter(
                        umap_2d[mask, 0],
                        umap_2d[mask, 1],
                        c=[cmap_lang(i)],
                        label=lang,
                        alpha=0.7,
                        s=5,
                        edgecolors="none",
                    )

                # Add labels and legend
                plt.title(
                    f"2D UMAP by Language ({position} embeddings), Fold {fold_number}"
                )
                plt.xlabel("UMAP Dimension 1")
                plt.ylabel("UMAP Dimension 2")

                # Add legend with larger dots - FIXED
                legend = plt.legend(
                    title="Language",
                    loc="best",
                    markerscale=10,  # Scale marker size in legend
                )

                plt.tight_layout()

                # Save plot
                output_file = os.path.join(
                    output_dir,
                    f"umap_2d_{position}_lang_fold_{fold_number}.{args.output_format}",
                )
                plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved plot to {output_file}")

                # Show plot if requested
                if args.show_plots:
                    plt.show()
                else:
                    plt.close()

            # If all_data is provided, create scatter plots colored by K-means clusters
            if all_data is not None:
                # Create plots for each requested k value
                for k in args.kmeans_clusters:
                    for position in args.embedding_positions:
                        if position not in embeddings:
                            continue

                        position_embeddings = embeddings[position]

                        if "umap_2d" not in position_embeddings:
                            continue

                        umap_2d = position_embeddings["umap_2d"]

                        # Get the exact name used in the clustering script
                        embed_name = f"{position}_umap_2d"

                        # FIXED: Ensure we're using the correct cluster labels
                        if (
                            embed_name in all_data
                            and k in all_data[embed_name]
                            and "cluster_labels" in all_data[embed_name][k]
                            and fold_number in all_data[embed_name][k]["cluster_labels"]
                        ):
                            cluster_labels = all_data[embed_name][k]["cluster_labels"][
                                fold_number
                            ]

                            # Add debugging info
                            if args.debug:
                                print(
                                    f"Fold {fold_number}, k={k}, position={position}, embedding={embed_name}"
                                )
                                print(
                                    f"Unique cluster labels: {np.unique(cluster_labels, return_counts=True)}"
                                )
                                print(f"Shape of UMAP data: {umap_2d.shape}")
                                print(
                                    f"Shape of cluster_labels: {cluster_labels.shape}"
                                )

                            # Verify that we have the correct number of labels
                            if len(cluster_labels) != len(umap_2d):
                                print(
                                    f"WARNING: Mismatch between cluster labels ({len(cluster_labels)}) "
                                    f"and UMAP points ({len(umap_2d)}) for {embed_name}, k={k}, fold={fold_number}"
                                )
                                continue

                            # Create a colormap for cluster labels
                            cmap_clusters = plt.cm.get_cmap("tab10", k)

                            # Create figure
                            plt.figure(figsize=(12, 10))

                            # FIXED: Create scatter plots for each cluster with proper legend scaling
                            scatter_objects = []
                            for cluster_id in range(k):
                                mask = cluster_labels == cluster_id
                                sc = plt.scatter(
                                    umap_2d[mask, 0],
                                    umap_2d[mask, 1],
                                    c=[cmap_clusters(cluster_id)],
                                    label=f"Cluster {cluster_id}",
                                    alpha=0.7,
                                    s=5,
                                    edgecolors="none",
                                )

                            plt.title(
                                f"2D UMAP with K-means (k={k}, {position} embeddings), Fold {fold_number}"
                            )
                            plt.xlabel("UMAP Dimension 1")
                            plt.ylabel("UMAP Dimension 2")

                            # FIXED: Create legend with proper marker scaling
                            legend = plt.legend(
                                title=f"K-means Clusters (k={k})",
                                loc="best",
                                bbox_to_anchor=(1.05, 1),
                                markerscale=10,  # Scale marker size in legend
                            )

                            plt.tight_layout()

                            # Save plot
                            output_file = os.path.join(
                                output_dir,
                                f"umap_2d_{position}_kmeans_k{k}_fold_{fold_number}.{args.output_format}",
                            )
                            plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
                            print(f"Saved plot to {output_file}")

                            if args.show_plots:
                                plt.show()
                            else:
                                plt.close()
                        else:
                            if args.debug:
                                print(
                                    f"DEBUG: Missing data for {embed_name}, k={k}, fold={fold_number}"
                                )
                                if embed_name in all_data:
                                    print(f"  - embed_name exists in all_data")
                                    if k in all_data[embed_name]:
                                        print(f"  - k exists in all_data[{embed_name}]")
                                        if "cluster_labels" in all_data[embed_name][k]:
                                            print(
                                                f"  - cluster_labels exists in all_data[{embed_name}][{k}]"
                                            )
                                            if (
                                                fold_number
                                                in all_data[embed_name][k][
                                                    "cluster_labels"
                                                ]
                                            ):
                                                print(
                                                    f"  - fold_number exists in all_data[{embed_name}][{k}]['cluster_labels']"
                                                )

                            print(
                                f"Warning: K-means cluster labels for k={k}, position={position}, fold={fold_number} not found"
                            )

        except Exception as e:
            print(f"Error processing UMAP file {fold_file}: {e}")
            import traceback

            traceback.print_exc()

    print("UMAP plotting complete.")


def main():
    """Main execution function."""
    print("Starting plot generation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics to plot: {args.metrics}")

    # Plot metrics
    print("\n=== PLOTTING CLUSTERING METRICS ===")

    # Load metrics data
    all_data = load_metrics_data()
    if all_data is None:
        print("Error loading metrics data, skipping metrics plots.")
    else:
        # Process metrics data
        processed_data = process_metrics_data(all_data)

        # Plot each requested metric
        for metric in args.metrics:
            plot_metric(metric, processed_data)

    # Plot UMAP visualizations if requested
    if args.plot_umap:
        print("\n=== PLOTTING UMAP VISUALIZATIONS ===")
        print(f"UMAP data directory: {umap_dir}")

        if not os.path.exists(umap_dir):
            print(f"Error: UMAP directory {umap_dir} does not exist.")
            print("Skipping UMAP plots.")
        else:
            plot_umap_scatter(umap_dir, all_data)

    print(f"\nPlot generation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
