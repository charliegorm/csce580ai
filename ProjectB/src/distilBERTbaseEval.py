import json, time, numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

def main():
    processedDir = Path("ProjectB/data/processed")
    reportsDir   = Path("ProjectB/reports") 
    figsDir = Path(os.path.join(reportsDir, "figures"))
    tablesDir = Path(os.path.join(reportsDir, "tables"))
    for d in [figsDir, tablesDir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # load tokenized test set
    testDs = load_from_disk(os.path.join(processedDir, "hf_test"))

    modelName = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(modelName)
    model = DistilBertForSequenceClassification.from_pretrained(modelName, num_labels=2)

    # no training, just evaluation of the base checkpoint
    trainer = Trainer(model=model, tokenizer=tokenizer)

    t0 = time.time()
    preds = trainer.predict(testDs)
    infer_time = time.time() - t0

    yProb = preds.predictions
    yPred = np.argmax(yProb, axis=-1)
    yTrue = preds.label_ids

    # accuracy
    acc = float((yPred == yTrue).mean())
    # precision, recall, F1 for positive class (label=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yTrue, yPred, average="binary", pos_label=1
    )

    out = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "test_infer_time_sec": float(infer_time),
    }
    with open(tablesDir / "distilbert_base_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    # confusion matrix plot
    cm = confusion_matrix(yTrue, yPred, labels=[0, 1])
    fig = plt.figure(figsize=(4.2, 3.6)); ax = plt.gca()
    im = ax.imshow(cm)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("DistilBERT (base) Confusion")
    fig.tight_layout()
    fig.savefig(figsDir / "distilbert_base_confusion.png", dpi=150)
    plt.close(fig)

    print("Saved base metrics/confusion.")

if __name__ == "__main__":
    main()
