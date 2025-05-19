import argparse
import json
import os
import pickle
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots
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
    default=[
        "raw",
        "pca_2d",
        "pca_4d",
        "pca_8d",
        "pca_16d",  # Added PCA types
        "pca_2d_umap_2d",
        "pca_4d_umap_2d",
        "pca_8d_umap_2d",
        "pca_16d_umap_2d",  # Added PCA+UMAP types
        "umap_2d",
        "umap_4d",
        "umap_8d",
        "umap_16d",
        "umap_32d",
    ],
    help="Embedding types to plot (default: raw, all PCA, all PCA+UMAP, all UMAP dimensions).",
)
parser.add_argument(
    "--output_format",
    type=str,
    default="pdf",
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
    "--plot_dimred",
    action="store_true",
    default=True,
    help="Generate 2D scatter plots colored by register (default: True).",
)
parser.add_argument(
    "--dimred_dir",
    type=str,
    default=None,
    help="Directory containing dimensionality reduction data. If None, will use base_dir/dimensionality_reduction_N_label_type.",
)
# NEW PARAMETER: Add cluster numbers to visualize in k-means plots
parser.add_argument(
    "--kmeans_clusters",
    nargs="+",
    type=int,
    default=[4, 8],
    help="K values to use for k-means cluster visualization (default: [4, 8, 25]).",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug printing for diagnosing issues.",
)
parser.add_argument(
    "--plot_groups",
    type=str,
    choices=["all", "pca", "umap", "pca_umap", "raw"],
    default="all",
    help="Which groups of dimensionality reductions to plot (default: all).",
)

parser.add_argument(
    "--interactive",
    action="store_true",
    default=True,
    help="Generate interactive plots using Plotly (default: True).",
)
parser.add_argument(
    "--max_hover_examples",
    type=int,
    default=100,
    help="Maximum number of examples to show in hover data (default: 100).",
)
# Add arguments for selected examples
parser.add_argument(
    "--save_selection",
    action="store_true",
    default=False,
    help="Enable saving selected points for later use in static plots.",
)
parser.add_argument(
    "--selection_output",
    type=str,
    default="selected_examples.json",
    help="File to save selected examples to.",
)

args = parser.parse_args()

# Setup directory paths
if args.ONLY_MAIN_LABEL:
    label_type = "main_labels_only"
else:
    label_type = "filtered_labels"

input_dir = os.path.join(args.base_dir, f"cluster_metrics_{args.N}_{label_type}")
output_dir = os.path.join(args.base_dir, f"plots_{args.N}_{label_type}")

# Set dimensionality reduction directory
if args.dimred_dir is None:
    dimred_dir = os.path.join(
        args.base_dir, f"dimensionality_reduction_{args.N}_{label_type}"
    )
else:
    dimred_dir = args.dimred_dir

# Create output directory
os.makedirs(output_dir, exist_ok=True)


# Filter embedding types based on the plot_groups parameter
def filter_embedding_types():
    if args.plot_groups == "all":
        return args.embedding_types

    filtered_types = []
    if args.plot_groups == "raw" or args.plot_groups == "all":
        filtered_types.append("raw")

    if args.plot_groups == "pca" or args.plot_groups == "all":
        filtered_types.extend(["pca_2d", "pca_4d", "pca_8d", "pca_16d"])

    if args.plot_groups == "umap" or args.plot_groups == "all":
        filtered_types.extend(["umap_2d", "umap_4d", "umap_8d", "umap_16d", "umap_32d"])

    if args.plot_groups == "pca_umap" or args.plot_groups == "all":
        filtered_types.extend(
            ["pca_2d_umap_2d", "pca_4d_umap_2d", "pca_8d_umap_2d", "pca_16d_umap_2d"]
        )

    return filtered_types


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

    # Get filtered embedding types
    embedding_types = filter_embedding_types()

    # Define color palettes for different embedding type groups
    raw_color = "#1f77b4"  # Blue for raw
    pca_palette = sns.color_palette("Reds_d", 4)  # Red palette for PCA
    umap_palette = sns.color_palette("Greens_d", 5)  # Green palette for UMAP
    pca_umap_palette = sns.color_palette("Purples_d", 4)  # Purple palette for PCA+UMAP

    # Create a combined color mapping
    embedding_colors = {"raw": raw_color}

    # Add PCA colors
    for i, dim in enumerate([2, 4, 8, 16]):
        embedding_colors[f"pca_{dim}d"] = pca_palette[i]

    # Add UMAP colors
    for i, dim in enumerate([2, 4, 8, 16, 32]):
        embedding_colors[f"umap_{dim}d"] = umap_palette[min(i, len(umap_palette) - 1)]

    # Add PCA+UMAP colors
    for i, dim in enumerate([2, 4, 8, 16]):
        embedding_colors[f"pca_{dim}d_umap_2d"] = pca_umap_palette[i]

    # Plot for each embedding position (first, half, last)
    for position in args.embedding_positions:
        plt.figure(figsize=(15, 10))

        # Group embedding types by category
        raw_types = [et for et in embedding_types if et == "raw"]
        pca_types = [et for et in embedding_types if "pca_" in et and "umap" not in et]
        umap_types = [et for et in embedding_types if "umap_" in et and "pca" not in et]
        pca_umap_types = [et for et in embedding_types if "pca_" in et and "umap" in et]

        # Sort dimension categories by dimensionality
        pca_types.sort(key=lambda x: int(x.split("_")[1][:-1]))
        umap_types.sort(key=lambda x: int(x.split("_")[1][:-1]))
        pca_umap_types.sort(key=lambda x: int(x.split("_")[1][:-1]))

        # All embedding types in correct order
        ordered_embed_types = raw_types + pca_types + pca_umap_types + umap_types

        # Plot each embedding type in order
        for embed_type in ordered_embed_types:
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
            if embed_type == "raw":  # For raw, use std
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
                color=embedding_colors.get(embed_type, "black"),
                linewidth=2,
                linestyle="-"
                if "umap" in embed_type
                else "--"
                if "pca" in embed_type and "umap" in embed_type
                else "-."
                if "pca" in embed_type
                else ":",
            )

            # Plot shaded area for range
            plt.fill_between(
                valid_k,
                valid_lower,
                valid_upper,
                alpha=0.1,
                color=embedding_colors.get(embed_type, "black"),
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

        # Improve legend with grouping
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Create legend with sections
        legend_elements = []

        # Add raw if present
        if any(label == "raw" for label in labels):
            legend_elements.append(by_label["raw"])
            legend_elements.append("raw")

        # Add PCA section if present
        pca_labels = [
            label for label in labels if "pca_" in label and "umap" not in label
        ]
        if pca_labels:
            legend_elements.append(
                plt.Line2D([0], [0], color="white", alpha=0.0)
            )  # spacer
            legend_elements.append("PCA")
            for label in sorted(pca_labels, key=lambda x: int(x.split("_")[1][:-1])):
                legend_elements.append(by_label[label])
                legend_elements.append(label)

        # Add PCA+UMAP section if present
        pca_umap_labels = [
            label for label in labels if "pca_" in label and "umap" in label
        ]
        if pca_umap_labels:
            legend_elements.append(
                plt.Line2D([0], [0], color="white", alpha=0.0)
            )  # spacer
            legend_elements.append("PCA+UMAP")
            for label in sorted(
                pca_umap_labels, key=lambda x: int(x.split("_")[1][:-1])
            ):
                legend_elements.append(by_label[label])
                legend_elements.append(label)

        # Add UMAP section if present
        umap_labels = [
            label for label in labels if "umap_" in label and "pca" not in label
        ]
        if umap_labels:
            legend_elements.append(
                plt.Line2D([0], [0], color="white", alpha=0.0)
            )  # spacer
            legend_elements.append("UMAP")
            for label in sorted(umap_labels, key=lambda x: int(x.split("_")[1][:-1])):
                legend_elements.append(by_label[label])
                legend_elements.append(label)

        # Add legend
        if legend_elements:
            plt.legend(
                legend_elements[::2],
                legend_elements[1::2],
                handler_map={str: plt.matplotlib.legend_handler.HandlerBase()},
                loc="best",
                bbox_to_anchor=(1.05, 1),
                fontsize=10,
            )

        plt.grid(True)
        plt.tight_layout()

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


# Modify the plot_2d_projections function
def plot_2d_projections(dimred_dir, all_data=None):
    """Create scatter plots of 2D projections colored by register and k-means clusters."""
    print(f"Plotting 2D projections from {dimred_dir}...")

    # Check if directory exists
    if not os.path.exists(dimred_dir):
        print(f"Error: Directory {dimred_dir} does not exist.")
        return

    # List fold files
    fold_files = [
        f
        for f in os.listdir(dimred_dir)
        if f.startswith("dim_reduction_fold_") and f.endswith(".pkl")
    ]
    fold_files.sort()

    if not fold_files:
        print(f"Error: No fold files found in {dimred_dir}")
        return

    print(f"Found {len(fold_files)} fold files to process")

    # Define which 2D embeddings to plot
    plot_keys = [
        "umap_2d",  # Original UMAP 2D
        "pca_2d",  # PCA 2D
        "pca_2d_umap_2d",  # UMAP 2D on PCA 2D
        "pca_4d_umap_2d",  # UMAP 2D on PCA 4D
        "pca_8d_umap_2d",  # UMAP 2D on PCA 8D
        "pca_16d_umap_2d",  # UMAP 2D on PCA 16D
    ]

    # Directory to save interactive plots
    interactive_output_dir = os.path.join(output_dir, "interactive")
    os.makedirs(interactive_output_dir, exist_ok=True)

    # Dictionary to store selections
    selected_examples = {}

    # Process each fold
    for fold_file in tqdm(fold_files, desc="Creating 2D projection plots"):
        fold_path = os.path.join(dimred_dir, fold_file)
        fold_number = fold_file.split("_")[3].split(".")[0]

        if fold_number != "10":
            continue

        try:
            # Load the fold data
            with open(fold_path, "rb") as f:
                fold_data = pickle.load(f)

            metadata = fold_data.get("metadata", [])
            embeddings = fold_data.get("embeddings", {})

            # Get actual text examples if available
            raw_texts = []
            for item in metadata:
                # Try to get the text content if available
                text = item.get("text", "")
                if not text:
                    text = item.get("content", "")  # Alternative field name
                if not text:
                    text = "Example text not available"  # Fallback

                # Truncate long texts for hover display
                if len(text) > 100:
                    text = text[:97] + "..."

                raw_texts.append(text)

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

            # Get language info for each point
            languages = np.array([item["lang"] for item in metadata])
            unique_languages = np.unique(languages)

            # Static plot functions remain unchanged...

            # Add interactive plotting using Plotly
            if args.interactive:
                for position in args.embedding_positions:
                    if position not in embeddings:
                        print(f"Warning: Position {position} not found in embeddings")
                        continue

                    position_embeddings = embeddings[position]

                    # Plot each 2D projection
                    for embed_key in plot_keys:
                        if embed_key not in position_embeddings:
                            if args.debug:
                                print(
                                    f"Warning: {embed_key} not found for {position} embeddings"
                                )
                            continue

                        proj_2d = position_embeddings[embed_key]

                        # Create additional information for hover
                        hover_data = []
                        for i, item in enumerate(
                            metadata[: min(len(metadata), len(proj_2d))]
                        ):
                            hover_info = {
                                "ID": item.get("id", f"sample_{i}"),
                                "Language": item.get("lang", "unknown"),
                                "Register": labels[i],
                                "Text": raw_texts[i]
                                if i < len(raw_texts)
                                else "No text available",
                            }
                            # Add any other useful metadata
                            for key, value in item.items():
                                if key not in hover_info and key not in [
                                    "text",
                                    "content",
                                ]:
                                    # Convert lists/dicts to string representation
                                    if isinstance(value, (list, dict)):
                                        value = str(value)
                                    hover_info[key] = value
                            hover_data.append(hover_info)

                        # 1. Interactive plot by register
                        fig = px.scatter(
                            x=proj_2d[:, 0],
                            y=proj_2d[:, 1],
                            color=labels,
                            title=f"{embed_key} Projection ({position} embeddings), Fold {fold_number}",
                            labels={"x": "Dimension 1", "y": "Dimension 2"},
                            color_discrete_map={
                                label: f"rgba({int(color[0] * 255)},{int(color[1] * 255)},{int(color[2] * 255)},{color[3]})"
                                for label, color in label_colors.items()
                            },
                            category_orders={"color": sorted(unique_labels)},
                            hover_data=hover_data,
                            custom_data=[
                                i for i in range(len(proj_2d))
                            ],  # For selection
                        )

                        # Improve the hover template
                        fig.update_traces(
                            hovertemplate="<b>Register:</b> %{color}<br>"
                            + "<b>ID:</b> %{customdata}<br>"
                            + "<b>Language:</b> %{hovername}<br>"
                            + "<b>Text:</b> %{hovertext}<br>"
                            + "<extra></extra>"
                        )

                        # Add selection functionality
                        if args.save_selection:
                            # Add JavaScript for selection
                            fig.update_layout(
                                clickmode="event+select",
                                # Add custom buttons
                                updatemenus=[
                                    dict(
                                        type="buttons",
                                        direction="right",
                                        buttons=[
                                            dict(
                                                args=[{"clickmode": "event+select"}],
                                                label="Enable Selection",
                                                method="relayout",
                                            ),
                                            dict(
                                                args=[{"clickmode": "none"}],
                                                label="Disable Selection",
                                                method="relayout",
                                            ),
                                        ],
                                        pad={"r": 10, "t": 10},
                                        showactive=True,
                                        x=0.1,
                                        xanchor="left",
                                        y=1.1,
                                        yanchor="top",
                                    ),
                                ],
                                margin=dict(l=40, r=40, t=60, b=40),
                            )

                            # Add annotation to explain selection
                            fig.add_annotation(
                                x=0.5,
                                y=1.05,
                                xref="paper",
                                yref="paper",
                                text="Click to select points. Selected points will be saved for static plots.",
                                showarrow=False,
                                font=dict(size=12),
                                align="center",
                            )

                            # Add a callback to save selected points (must be added externally)
                            selected_examples_key = (
                                f"{embed_key}_{position}_register_fold_{fold_number}"
                            )
                            selected_examples[selected_examples_key] = {
                                "embed_key": embed_key,
                                "position": position,
                                "fold_number": fold_number,
                                "projection": proj_2d.tolist(),
                                "labels": labels.tolist(),
                                "metadata": [
                                    {
                                        "id": item.get("id", f"sample_{i}"),
                                        "lang": item.get("lang", "unknown"),
                                        "register": labels[i],
                                        "text": raw_texts[i]
                                        if i < len(raw_texts)
                                        else "No text available",
                                    }
                                    for i, item in enumerate(
                                        metadata[: min(len(metadata), len(proj_2d))]
                                    )
                                ],
                                "selected_indices": [],  # To be filled by user interaction
                            }

                        # Save interactive plot as HTML
                        output_file = os.path.join(
                            interactive_output_dir,
                            f"{embed_key}_{position}_register_fold_{fold_number}_interactive.html",
                        )

                        # Add JavaScript for selection export if requested
                        if args.save_selection:
                            # Add JavaScript to handle selection and export
                            selection_js = f"""
                            <script>
                                var selectedPoints = [];
                                var plotKey = "{selected_examples_key}";
                                
                                document.addEventListener('DOMContentLoaded', function() {{
                                    var plotEl = document.querySelector('.plotly-graph-div');
                                    plotEl.on('plotly_selected', function(eventData) {{
                                        if (eventData && eventData.points) {{
                                            selectedPoints = eventData.points.map(pt => pt.customdata);
                                            console.log('Selected points:', selectedPoints);
                                            
                                            // Update display
                                            document.getElementById('selected-count').innerText = selectedPoints.length;
                                        }}
                                    }});
                                    
                                    // Add export button
                                    var exportBtn = document.createElement('button');
                                    exportBtn.innerText = 'Export Selection';
                                    exportBtn.style.position = 'absolute';
                                    exportBtn.style.top = '10px';
                                    exportBtn.style.right = '10px';
                                    exportBtn.style.zIndex = 999;
                                    exportBtn.onclick = function() {{
                                        var selection = {{
                                            plot_key: plotKey,
                                            selected_indices: selectedPoints
                                        }};
                                        
                                        // Create download link
                                        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(selection));
                                        var downloadAnchorNode = document.createElement('a');
                                        downloadAnchorNode.setAttribute("href", dataStr);
                                        downloadAnchorNode.setAttribute("download", "selection_" + plotKey + ".json");
                                        document.body.appendChild(downloadAnchorNode);
                                        downloadAnchorNode.click();
                                        downloadAnchorNode.remove();
                                    }};
                                    
                                    // Add status display
                                    var statusDiv = document.createElement('div');
                                    statusDiv.style.position = 'absolute';
                                    statusDiv.style.top = '40px';
                                    statusDiv.style.right = '10px';
                                    statusDiv.style.zIndex = 999;
                                    statusDiv.innerHTML = 'Selected: <span id="selected-count">0</span> points';
                                    
                                    // Add to document
                                    document.body.appendChild(exportBtn);
                                    document.body.appendChild(statusDiv);
                                }});
                            </script>
                            """

                            # Save with added JavaScript
                            with open(output_file, "w") as f:
                                html_content = pio.to_html(
                                    fig, include_plotlyjs=True, full_html=True
                                )
                                html_content = html_content.replace(
                                    "</body>", f"{selection_js}</body>"
                                )
                                f.write(html_content)
                        else:
                            # Save without selection JavaScript
                            fig.write_html(output_file)

                        print(f"Saved interactive plot to {output_file}")

                        # Open in browser for the first plot
                        if (
                            position == args.embedding_positions[0]
                            and embed_key == plot_keys[0]
                        ):
                            try:
                                webbrowser.open(
                                    "file://" + os.path.abspath(output_file)
                                )
                                print(f"Opening {output_file} in browser")
                            except Exception as e:
                                print(f"Could not open browser: {e}")

                        # 2. Interactive plot by language
                        fig = px.scatter(
                            x=proj_2d[:, 0],
                            y=proj_2d[:, 1],
                            color=languages,
                            title=f"{embed_key} by Language ({position} embeddings), Fold {fold_number}",
                            labels={"x": "Dimension 1", "y": "Dimension 2"},
                            category_orders={"color": sorted(unique_languages)},
                            hover_data=hover_data,
                        )

                        # Improve the hover template
                        fig.update_traces(
                            hovertemplate="<b>Language:</b> %{color}<br>"
                            + "<b>Register:</b> %{hovername}<br>"
                            + "<b>Text:</b> %{hovertext}<br>"
                            + "<extra></extra>"
                        )

                        # Save interactive plot as HTML
                        output_file = os.path.join(
                            interactive_output_dir,
                            f"{embed_key}_{position}_lang_fold_{fold_number}_interactive.html",
                        )
                        fig.write_html(output_file)
                        print(f"Saved interactive plot to {output_file}")

            # If all_data is provided, create scatter plots colored by K-means clusters
            # This section is unchanged...

        except Exception as e:
            print(f"Error processing fold file {fold_file}: {e}")
            import traceback

            traceback.print_exc()

    # Save selected examples if any were created
    if args.save_selection and selected_examples:
        selection_path = os.path.join(output_dir, args.selection_output)
        with open(selection_path, "w") as f:
            json.dump(selected_examples, f, indent=2)
        print(f"Saved selection template to {selection_path}")

    print("2D projection plotting complete.")


def main():
    """Main execution function."""
    print("Starting plot generation...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics to plot: {args.metrics}")

    # Print info about which embedding types we're using
    embedding_types = filter_embedding_types()
    print(f"Using embedding types: {embedding_types}")

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

    # Plot 2D visualizations if requested
    if args.plot_dimred:
        print("\n=== PLOTTING 2D VISUALIZATIONS ===")
        print(f"Data directory: {dimred_dir}")

        if not os.path.exists(dimred_dir):
            print(f"Error: Directory {dimred_dir} does not exist.")
            print("Skipping 2D plots.")
        else:
            plot_2d_projections(dimred_dir, all_data)

    print(f"\nPlot generation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
