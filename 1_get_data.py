import os
import pandas as pd
import numpy as np
import ast
import glob
from tqdm import tqdm

# Define constants
BASE_PATH = (
    "/scratch/project_462000353/amanda/register-clustering/data/model_embeds/hplt/"
)
COLUMNS_TO_KEEP = ["lang", "embed_last", "preds_0.4"]
OUTPUT_DIR = "processed_data_all"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each fold directory
for fold_dir in glob.glob(os.path.join(BASE_PATH, "bge-m3-fold-*")):
    # Extract fold number from directory name
    fold_num = int(fold_dir.split("-")[-1])
    print(f"Processing fold {fold_num}...")

    # Initialize an empty list to store dataframes for this fold
    fold_dfs = []

    # Process each TSV file in the current fold directory
    for tsv_file in glob.glob(os.path.join(fold_dir, "*.tsv")):
        print(f"Processing file: {tsv_file}")

        # Read the TSV file in chunks to handle large files
        chunk_size = 100000  # Adjust based on available memory

        for chunk in tqdm(
            pd.read_csv(
                tsv_file, sep="\t", usecols=COLUMNS_TO_KEEP, chunksize=chunk_size
            )
        ):
            # Add fold column
            chunk["fold"] = fold_num

            # Parse preds_0.4 column
            chunk["preds_0.4"] = chunk["preds_0.4"].apply(lambda x: ast.literal_eval(x))

            # Parse embedding column
            chunk["embed_last"] = chunk["embed_last"].apply(
                lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
            )

            # Append chunk to the list
            if not chunk.empty:
                fold_dfs.append(chunk)

    # Combine all chunks for this fold
    if fold_dfs:
        fold_df = pd.concat(fold_dfs, ignore_index=True)

        # Save to pickle file
        output_file = os.path.join(OUTPUT_DIR, f"data_{fold_num}.pkl")
        print(f"Saving fold {fold_num} to {output_file}")
        fold_df.to_pickle(output_file)

        # Free memory
        del fold_df, fold_dfs
