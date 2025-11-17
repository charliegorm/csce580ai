import pandas as pd
from pathlib import Path
import os
from datasets import load_dataset


def main():
    rawDir = Path("ProjectB/data/raw")
    rawDir.mkdir(parents=True, exist_ok=True)
    localPath = os.path.join(rawDir, "IMDB_Dataset.csv")
    
    # tried using load_dataset (provided by Kaggle) but was outdated, tried dataset_load (kaggle) but was giving errors for reading,
    # tried to just download and then let pandas read instead, more issues with encoding, using HuggingFace's datasets instead 
    # (identical dataset after my transformations)
    print("Loading IMDB dataset from Hugging Face …")
    ds = load_dataset("imdb")
    
    # combine train + test splits and save as Kaggle-style CSV (from HuggingFace datasets)
    dfTrain = ds["train"].to_pandas()
    dfTest  = ds["test"].to_pandas()
    df = pd.concat([dfTrain, dfTest], ignore_index=True)
    df.rename(columns={"text": "review", "label": "sentiment"}, inplace=True)
    df["sentiment"] = df["sentiment"].map({1: "positive", 0: "negative"})
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(3))

    # dataset overview
    print("\nSentiment balance:")
    print(df["sentiment"].value_counts())

    # raw CSV for later scripts
    df.to_csv(localPath, index=False)
    print(f"\nSaved dataset to {localPath}")

if __name__ == "__main__":
    main()
