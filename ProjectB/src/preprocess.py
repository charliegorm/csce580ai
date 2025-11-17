import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib, scipy.sparse as sp

from datasets import Dataset
from transformers import DistilBertTokenizerFast


def main():
    rawDir = Path("ProjectB/data/raw")
    interimDir = Path("ProjectB/data/interim")
    processedDir = Path("ProjectB/data/processed")
    interimDir.mkdir(parents=True, exist_ok=True)
    processedDir.mkdir(parents=True, exist_ok=True)

    rawCsvPath = os.path.join(rawDir, "IMDB_Dataset.csv")

    # loading csv + basic prepping
    print(f"Loading raw CSV from {rawCsvPath} ...")
    df = pd.read_csv(rawCsvPath)

    # map sentiment to numeric labels for modeling (could've kept huggingface dataset the same )
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # split (80/20 stratified) for DistilBERT and logistic regression training/testing, will explain in report later as well
    # stratify ensures there is equal pos/neg reviews in each split
    print("Splitting train/test (stratified, 80/20) ...")
    trainDf, testDf = train_test_split(
        df[["review", "label"]], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # saving interim CSVs
    trainCsv = os.path.join(interimDir, "train.csv") 
    testCsv  = os.path.join(interimDir, "test.csv")
    trainDf.to_csv(trainCsv, index=False)
    testDf.to_csv(testCsv, index=False)
    print(f"Saved splits -> {trainCsv} ({len(trainDf)} rows), {testCsv} ({len(testDf)} rows)")

    # TF-IDF, turning words -> weighted numeric features for the classical models to classify the reviews
    print("Building TF-IDF features (max_features=5000, ngram 1-2, english stopwords) ...") # ignores common filler words (the,and,is)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    XTrain = tfidf.fit_transform(trainDf["review"]) # each row corresponds to one review, each col corresponds to words/phrases 
    # w the value being the weight of how important the word/term is in that review, compared to other reviews
    XTest  = tfidf.transform(testDf["review"]) # since this is test data, not transformed but applies what was learned to allow
    # the model to fairly evaluate on unseen data
    
    # saving TF-IDF artifacts so i can reuse in training scripts later
    joblib.dump(tfidf, processedDir / "tfidf.pkl") # TF-IDF vectorizer object
    sp.save_npz(processedDir / "X_train.npz", XTrain) # sparse feature matrices for train,test
    sp.save_npz(processedDir / "X_test.npz",  XTest)
    np.save(processedDir / "y_train.npy", trainDf["label"].to_numpy()) # corresponding labels 0-neg 1-pos train,test
    np.save(processedDir / "y_test.npy",  testDf["label"].to_numpy())
    print(f"Saved TF-IDF + labels to {processedDir}")

    # DistilBERT tokenization, similar to before, taking the raw strings -> numbers (so model can understand),
    # transformers (like DistilBERT) use the unique ids to represent the words
    print("Tokenizing with DistilBERT tokenizer (max_length=256) ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # cutting long reviews and padding short ones all sequences need to be the same length, 
    # actual tokens get attention_mask = 1 and padding = 0
    def tokenizeBatch(batch):
        return tokenizer(
            batch["review"], truncation=True, padding="max_length", max_length=256
        )

    # create huggingface (HF) Datasets from pandas, just renaming label -> labels to match HF conventions
    hfTrain = Dataset.from_pandas(trainDf.rename(columns={"label": "labels"}))
    hfTest  = Dataset.from_pandas(testDf.rename(columns={"label": "labels"}))

    # applying tokenizer to each row, adds input_ids () and attention_mask () columns 
    hfTrain = hfTrain.map(tokenizeBatch, batched=True) 
    hfTest  = hfTest.map(tokenizeBatch, batched=True)

    # drop the auto added pandas index column if present
    for col in ["__index_level_0__"]:
        if col in hfTrain.column_names: hfTrain = hfTrain.remove_columns(col)
        if col in hfTest.column_names:  hfTest  = hfTest.remove_columns(col)

    hfTrainPath = processedDir / "hf_train"
    hfTestPath  = processedDir / "hf_test"
    hfTrain.save_to_disk(str(hfTrainPath))
    hfTest.save_to_disk(str(hfTestPath))
    print(f"Saved tokenized HF datasets -> {hfTrainPath}, {hfTestPath}")

    # quick summaries of what was saved / done 
    print("\n------- Preprocessing Summary -------")
    print(f"Train size: {len(trainDf)} | Test size: {len(testDf)}")
    print("Label balance (train):")
    print(trainDf["label"].value_counts())
    print("Label balance (test):")
    print(testDf["label"].value_counts())
    print("Sample tokenized input_ids length:", len(hfTrain[0]["input_ids"]))
    print("Done.")


if __name__ == "__main__":
    main()
