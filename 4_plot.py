import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap

# Add these imports for UMAP plotting
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
# Add new arguments for UMAP visualization
parser.add_argument(
    "--plot_umap",
    action="store_true",
    help="Generate UMAP visualizations for 2D embeddings.",
)
parser.add_argument(
    "--embeddings_dir",
    type=str,
    default=None,
    help="Directory containing embeddings data (required for UMAP plotting).",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=5000,
    help="Maximum number of samples to use for UMAP visualization.",
)
args = parser.parse_args()

# Setup directory paths
if args.ONLY_MAIN_LABEL:
    label_type = "main_labels_only"
else:
    label_type = "filtered_labels"

input_dir = os.path.join(args.base_dir, f"cluster_metrics_{args.N}_{label_type}")
output_dir = os.path.join(args.base_dir, f"plots_{args.N}_{label_type}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)
if args.plot_umap:
    umap_output_dir = os.path.join(output_dir, "umap_plots")
    os.makedirs(umap_output_dir, exist_ok=True)


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
                alpha=0.2,
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


def load_embeddings_data(fold_num=0):
    """
    Load embeddings and labels from the embeddings directory for a specific fold.
    Returns embeddings, labels, and language data.
    """
    if not args.embeddings_dir:
        print("Error: embeddings_dir must be specified for UMAP plotting.")
        return None, None, None

    print(f"Loading embeddings data from {args.embeddings_dir}...")

    # Check if embeddings directory exists
    if not os.path.exists(args.embeddings_dir):
        print(f"Error: Embeddings directory {args.embeddings_dir} does not exist.")
        return None, None, None

    # Find the embeddings file for the specified fold
    embedding_files = [
        f
        for f in os.listdir(args.embeddings_dir)
        if f.startswith(f"fold_{fold_num}_") and f.endswith(".pkl")
    ]

    if not embedding_files:
        print(f"Error: No embedding files found for fold {fold_num}")
        return None, None, None

    # Load the first matching file
    embedding_path = os.path.join(args.embeddings_dir, embedding_files[0])
    with open(embedding_path, "rb") as f:
        data = pickle.load(f)

    # Extract embeddings, labels, and languages
    embeddings = data.get("embeddings", {})
    labels = data.get("labels", [])
    languages = data.get("languages", [])

    return embeddings, labels, languages


def plot_umap_visualization():
    """
    Create UMAP visualizations for 2D embeddings.
    This plots actual data points in 2D space with colors based on labels.
    """
    if not args.plot_umap:
        return

    print("Generating UMAP visualizations...")

    # Load embeddings data (using fold 0 by default)
    embeddings, labels, languages = load_embeddings_data(fold_num=0)
    if embeddings is None:
        return

    # Process each embedding position
    for position in args.embedding_positions:
        # We'll only visualize the 2D UMAP embeddings
        embed_name = f"{position}_umap_2d"

        if embed_name not in embeddings:
            print(f"Warning: {embed_name} not found in embeddings data, skipping.")
            continue

        embed_data = embeddings[embed_name]

        # Subsample if needed
        if len(embed_data) > args.max_samples:
            indices = np.random.choice(len(embed_data), args.max_samples, replace=False)
            embed_data = embed_data[indices]
            sampled_labels = [labels[i] for i in indices]
            if languages:
                sampled_languages = [languages[i] for i in indices]
        else:
            sampled_labels = labels
            if languages:
                sampled_languages = languages

        # Convert labels to numeric form if they're strings
        unique_labels = sorted(set(sampled_labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_id[label] for label in sampled_labels])

        # 1. Plot by labels
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embed_data[:, 0],
            embed_data[:, 1],
            c=numeric_labels,
            cmap="tab20",
            alpha=0.7,
            s=30,
        )

        # Add a legend
        if len(unique_labels) <= 20:  # Only show legend if not too many labels
            legend1 = plt.legend(
                *scatter.legend_elements(),
                loc="upper right",
                title="Labels",
                bbox_to_anchor=(1.15, 1),
            )
            plt.gca().add_artist(legend1)

        plt.title(f"UMAP 2D Visualization by Label ({position} embeddings)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(
            umap_output_dir, f"umap_2d_{position}_by_label.{args.output_format}"
        )
        plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved UMAP plot to {output_file}")

        if args.show_plots:
            plt.show()
        else:
            plt.close()

        # 2. If language data is available, plot by language
        if languages:
            plt.figure(figsize=(12, 10))

            # Convert languages to numeric form
            unique_langs = sorted(set(sampled_languages))
            lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
            numeric_langs = np.array([lang_to_id[lang] for lang in sampled_languages])

            scatter = plt.scatter(
                embed_data[:, 0],
                embed_data[:, 1],
                c=numeric_langs,
                cmap="tab10",
                alpha=0.7,
                s=30,
            )

            # Add a legend
            if len(unique_langs) <= 10:  # Only show legend if not too many languages
                legend1 = plt.legend(
                    *scatter.legend_elements(),
                    loc="upper right",
                    title="Languages",
                    bbox_to_anchor=(1.15, 1),
                )
                plt.gca().add_artist(legend1)

            plt.title(f"UMAP 2D Visualization by Language ({position} embeddings)")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            plt.tight_layout()

            # Save plot
            output_file = os.path.join(
                umap_output_dir, f"umap_2d_{position}_by_language.{args.output_format}"
            )
            plt.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved UMAP plot to {output_file}")

            if args.show_plots:
                plt.show()
            else:
                plt.close()


def main():
    """Main execution function."""
    print("Starting plot generation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics to plot: {args.metrics}")

    # Load metrics data
    all_data = load_metrics_data()
    if all_data is None:
        return

    # Process metrics data
    processed_data = process_metrics_data(all_data)

    # Plot each requested metric
    for metric in args.metrics:
        plot_metric(metric, processed_data)

    # Plot UMAP visualizations if requested
    if args.plot_umap:
        if not args.embeddings_dir:
            print("Warning: --embeddings_dir must be specified for UMAP plotting.")
        else:
            plot_umap_visualization()

    print(f"Plot generation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
