"""
VERITAS Data Pipeline
Downloads FEVER, LIAR-PLUS, AVeriTeC datasets
and builds the ChromaDB knowledge base.
"""

from datasets import load_dataset
import pandas as pd
import os
import json

DATA_PATH = "./data"
os.makedirs(f"{DATA_PATH}/fever", exist_ok=True)
os.makedirs(f"{DATA_PATH}/liar", exist_ok=True)
os.makedirs(f"{DATA_PATH}/averitec", exist_ok=True)


def download_fever():
    print("\n[1/3] Downloading FEVER...")
    dataset = load_dataset("fever", "v1.0")
    
    train = dataset["train"].to_pandas()
    dev = dataset["paper_dev"].to_pandas()
    
    train.to_csv(f"{DATA_PATH}/fever/train.csv", index=False)
    dev.to_csv(f"{DATA_PATH}/fever/dev.csv", index=False)
    
    print(f"    Train: {len(train):,} claims")
    print(f"    Dev:   {len(dev):,} claims")
    print(f"    Labels: {train['label'].value_counts().to_dict()}")
    print("    FEVER OK")
    return train, dev


def download_liar():
    print("\n[2/3] Downloading LIAR-PLUS...")
    dataset = load_dataset("liar")
    
    train = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()
    
    train.to_csv(f"{DATA_PATH}/liar/train.csv", index=False)
    test.to_csv(f"{DATA_PATH}/liar/test.csv", index=False)
    
    print(f"    Train: {len(train):,} claims")
    print(f"    Test:  {len(test):,} claims")
    print(f"    Labels: {train['label'].value_counts().to_dict()}")
    print("    LIAR OK")
    return train, test


def download_averitec():
    print("\n[3/3] Downloading AVeriTeC...")
    try:
        dataset = load_dataset("AVeriTeC/AVeriTeC")
        
        train = dataset["train"].to_pandas()
        dev = dataset["validation"].to_pandas()
        
        train.to_csv(f"{DATA_PATH}/averitec/train.csv", index=False)
        dev.to_csv(f"{DATA_PATH}/averitec/dev.csv", index=False)
        
        print(f"    Train: {len(train):,} claims")
        print(f"    Dev:   {len(dev):,} claims")
        print("    AVeriTeC OK")
        return train, dev
    except Exception as e:
        print(f"    AVeriTeC failed: {e}")
        print("    Skipping — will retry manually")
        return None, None


if __name__ == "__main__":
    print("=" * 50)
    print("VERITAS Dataset Downloader")
    print("=" * 50)
    
    fever_train, fever_dev = download_fever()
    liar_train, liar_test = download_liar()
    averitec_train, averitec_dev = download_averitec()
    
    print("\n" + "=" * 50)
    print("All downloads complete.")
    print(f"Data saved to: {DATA_PATH}/")
    print("=" * 50)
