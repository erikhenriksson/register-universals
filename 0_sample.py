import os
import csv
import ast
import pickle
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import sys

# Increase CSV field size limit to maximum integer size
csv.field_size_limit(2147483647)  # Use a large but safe integer value

# Parse command line arguments
parser = argparse.ArgumentParser(description="Sample data from TSV files.")
parser.add_argument(
    "--N", type=int, default=1000, help="Number of samples per label per language."
)
parser.add_argument(
    "--ONLY_MAIN_LABEL",
    action="store_true",
    help="If set, only sample rows with a single main label.",
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="/scratch/project_462000353/amanda/register-clustering/data/model_embeds/hplt/",
    help="Base directory containing fold directories.",
)
args = parser.parse_args()

N = args.N
ONLY_MAIN_LABEL = args.ONLY_MAIN_LABEL
base_dir = args.base_dir

# Automatically determine number of worker processes (90% of available CPUs)
available_cpus = mp.cpu_count()
num_workers = max(1, int(available_cpus * 0.9))
print(f"Detected {available_cpus} CPUs, using {num_workers} worker processes (90%)")

# Output directory with clean, readable name
if ONLY_MAIN_LABEL:
    label_type = "main_labels_only"
else:
    label_type = "filtered_labels"

output_dir = f"sampled_data_{N}_{label_type}"
os.makedirs(output_dir, exist_ok=True)

print(f"Will save data to: {output_dir}")

# Languages to process
languages = ["en", "fr", "ur", "zh"]

# Find all fold directories
fold_dirs = [d for d in os.listdir(base_dir) if d.startswith("bge-m3-fold-")]
fold_dirs.sort()  # Sort to process in order


# Function to process a single fold
def process_fold(fold_dir, base_dir, languages, N, ONLY_MAIN_LABEL, output_dir):
    print(f"Processing {fold_dir}...")
    fold_path = os.path.join(base_dir, fold_dir)

    # Initialize data structure for this fold
    fold_data = []

    # Dictionary to track statistics
    fold_stats = {lang: {} for lang in languages}

    # Process each language file
    for lang in languages:
        tsv_file = os.path.join(fold_path, f"{lang}_embeds.tsv")
        if not os.path.exists(tsv_file):
            print(f"Warning: {tsv_file} does not exist. Skipping.")
            continue

        # Read TSV file
        rows = []
        with open(tsv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)

        print(f"  Processing {lang} file with {len(rows)} rows")

        # Group rows by label for sampling
        label_groups = defaultdict(list)

        for row in tqdm(rows, desc=f"Filtering {lang} rows", leave=False):
            # Parse the preds_0.4 column (it's a string representation of a list)
            try:
                preds_0_4 = ast.literal_eval(row["preds_0.4"])
            except (SyntaxError, ValueError):
                print(
                    f"Warning: Could not parse preds_0.4 for row with id {row.get('id', 'unknown')}. Skipping."
                )
                continue

            # Apply filtering based on ONLY_MAIN_LABEL
            if ONLY_MAIN_LABEL:
                # Only keep examples with a single uppercase (main) label, excluding "MT"
                main_labels = [label for label in preds_0_4 if label.isupper()]
                if len(main_labels) == 1 and main_labels[0] != "MT":
                    label = main_labels[0]
                    label_groups[label].append(row)
            else:
                # Remove lowercase labels and "MT", then check if there's only one left
                filtered_labels = [
                    label
                    for label in preds_0_4
                    if not label.islower() and label != "MT"
                ]
                if len(filtered_labels) == 1:
                    label = filtered_labels[0]
                    label_groups[label].append(row)

        # Sample N rows for each label (or all if less than N)
        for label, group in label_groups.items():
            sampled = random.sample(group, min(N, len(group)))

            # Track statistics
            fold_stats[lang][label] = len(sampled)

            for item in sampled:
                # Parse and process the embedding columns
                try:
                    embed_first = np.array(
                        ast.literal_eval(item["embed_first"])[0], dtype=np.float32
                    )
                    embed_half = np.array(
                        ast.literal_eval(item["embed_half"])[0], dtype=np.float32
                    )
                    embed_last = np.array(
                        ast.literal_eval(item["embed_last"])[0], dtype=np.float32
                    )

                    # Create processed row
                    processed_item = {
                        "lang": item["lang"],
                        "text": item["text"],
                        "preds_0.4": ast.literal_eval(item["preds_0.4"]),
                        "embed_first": embed_first,
                        "embed_half": embed_half,
                        "embed_last": embed_last,
                    }

                    fold_data.append(processed_item)
                except (SyntaxError, ValueError, IndexError) as e:
                    print(
                        f"Warning: Error processing embeddings for row with id {item.get('id', 'unknown')}: {e}. Skipping."
                    )
                    continue

    # Save the fold data to a pickle file
    fold_number = fold_dir.split("-")[-1]
    output_file = os.path.join(output_dir, f"fold_{fold_number}.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(fold_data, f)

    print(f"Saved {len(fold_data)} samples for {fold_dir} to {output_file}")
    return (
        fold_dir,
        len(fold_data),
        fold_stats,
    )  # Return fold name, count, and statistics


# Set up the process pool and run the processing
process_fold_partial = partial(
    process_fold,
    base_dir=base_dir,
    languages=languages,
    N=N,
    ONLY_MAIN_LABEL=ONLY_MAIN_LABEL,
    output_dir=output_dir,
)

# Process folds in parallel if num_workers > 1, otherwise sequentially
if num_workers > 1:
    print(f"Using {num_workers} worker processes for parallel processing")
    # Using imap instead of map to get better progress reporting
    with mp.Pool(processes=num_workers) as pool:
        results = []
        # Use imap for better progress monitoring with tqdm
        for result in tqdm(
            pool.imap_unordered(process_fold_partial, fold_dirs),
            total=len(fold_dirs),
            desc="Processing folds",
        ):
            results.append(result)

    # Print summary of processed folds
    print("\nProcessing summary:")
    for fold_dir, count, stats in results:
        print(f"  {fold_dir}: {count} samples")

    # Print detailed statistics
    print("\nDetailed statistics (samples per label per language):")
    for fold_dir, count, stats in results:
        fold_number = fold_dir.split("-")[-1]
        print(f"\nFold {fold_number} statistics:")

        # Find all unique labels across all languages
        all_labels = set()
        for lang in languages:
            if lang in stats:
                all_labels.update(stats[lang].keys())

        # Sort labels for consistent output
        sorted_labels = sorted(list(all_labels))

        # Print header
        print(f"{'Language':<10}", end="")
        for label in sorted_labels:
            print(f"{label:<10}", end="")
        print("Total")

        # Print counts for each language
        lang_totals = {}
        for lang in languages:
            if lang in stats:
                print(f"{lang:<10}", end="")
                lang_total = 0
                for label in sorted_labels:
                    count = stats[lang].get(label, 0)
                    lang_total += count
                    print(f"{count:<10}", end="")
                print(f"{lang_total}")
                lang_totals[lang] = lang_total

        # Print label totals
        print(f"{'Total':<10}", end="")
        grand_total = 0
        for label in sorted_labels:
            label_total = sum(
                stats[lang].get(label, 0) for lang in languages if lang in stats
            )
            grand_total += label_total
            print(f"{label_total:<10}", end="")
        print(f"{grand_total}")
else:
    print("Processing sequentially (no multiprocessing)")
    results = []
    for fold_dir in tqdm(fold_dirs, desc="Processing folds"):
        results.append(process_fold_partial(fold_dir))

    # Print summary of processed folds
    print("\nProcessing summary:")
    for fold_dir, count, stats in results:
        print(f"  {fold_dir}: {count} samples")

    # Print detailed statistics
    print("\nDetailed statistics (samples per label per language):")
    for fold_dir, count, stats in results:
        fold_number = fold_dir.split("-")[-1]
        print(f"\nFold {fold_number} statistics:")

        # Find all unique labels across all languages
        all_labels = set()
        for lang in languages:
            if lang in stats:
                all_labels.update(stats[lang].keys())

        # Sort labels for consistent output
        sorted_labels = sorted(list(all_labels))

        # Print header
        print(f"{'Language':<10}", end="")
        for label in sorted_labels:
            print(f"{label:<10}", end="")
        print("Total")

        # Print counts for each language
        lang_totals = {}
        for lang in languages:
            if lang in stats:
                print(f"{lang:<10}", end="")
                lang_total = 0
                for label in sorted_labels:
                    count = stats[lang].get(label, 0)
                    lang_total += count
                    print(f"{count:<10}", end="")
                print(f"{lang_total}")
                lang_totals[lang] = lang_total

        # Print label totals
        print(f"{'Total':<10}", end="")
        grand_total = 0
        for label in sorted_labels:
            label_total = sum(
                stats[lang].get(label, 0) for lang in languages if lang in stats
            )
            grand_total += label_total
            print(f"{label_total:<10}", end="")
        print(f"{grand_total}")

print(f"Sampling complete. Data saved to {output_dir}/")
