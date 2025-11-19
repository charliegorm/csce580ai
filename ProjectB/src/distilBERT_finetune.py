import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_from_disk
from transformers import (DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import load_from_disk, ClassLabel

# builds training args that work despite the build of transformers, tries modern args first then falls back,
# having issues with versions
def make_training_args(output_dir: str):
    try:
        # modern signature (supports evaluation/save/logging strategies)
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1, # changed from 2(used 2 on first implementation, takes too long) to 1 to speed up training
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            evaluation_strategy="no", # disabled per-epoch eval to save time, will still eval after
            save_strategy="no", # as above
            logging_strategy="steps",
            logging_steps=100,
            report_to=[]
        )
    except TypeError:
        # fallback, older builds that dont accept strategy kwargs
        # still pass valDs to Trainer just won’t auto eval each epoch
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1, # for consistency as above
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=100,
            report_to=[]
        )


# similar to the log reg training file's helper
def computeMetrics(evalPred):
    logits, labels = evalPred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# similar to the log reg training file's helper
def saveConfusion(figPath: Path, yTrue, yPred, title: str):
    cm = confusion_matrix(yTrue, yPred, labels=[0,1])
    fig = plt.figure(figsize=(4.2,3.6)); ax = plt.gca()
    im = ax.imshow(cm); ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["neg","pos"]); ax.set_yticklabels(["neg","pos"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(figPath, dpi=150); plt.close(fig)

def main():
    processedDir = Path("ProjectB/data/processed")
    modelsDir = Path("ProjectB/models/distilbert")
    reportsDir = Path("ProjectB/reports")
    figsDir = Path(os.path.join(reportsDir, "figures"))
    tablesDir = Path(os.path.join(reportsDir, "tables"))
    for d in [modelsDir, figsDir, tablesDir]: 
        d.mkdir(parents=True, exist_ok=True)

    print("Loading tokenized datasets from disk ...")
    trainAll = load_from_disk(str(processedDir / "hf_train"))
    testDs = load_from_disk(str(processedDir / "hf_test"))
    
    # making 'labels' a ClassLabel so we can stratify
    if not isinstance(trainAll.features["labels"], ClassLabel):
        trainAll = trainAll.cast_column("labels", ClassLabel(names=["negative", "positive"])) # 0 -> negative, 1 -> positive

    # further split for validation from the initial 80/20 split, 80 train 20 test, 10% of 80% -> 8% for validation
    print("Creating validation split (10% of train) ...")
    split = trainAll.train_test_split(test_size=0.1, seed=42, stratify_by_column="labels")
    trainDs, valDs = split["train"], split["test"]

    # subsample datasets to make training much faster
    trainDs = trainDs.shuffle(seed=42).select(range(10000)) # 10k training examples
    valDs   = valDs.shuffle(seed=42).select(range(2000)) # 2k validation examples

    modelName = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(modelName)
    model = DistilBertForSequenceClassification.from_pretrained(modelName, num_labels=2)

    # freezing encoder layers so only the classifier head is trained
    for p in model.distilbert.parameters():
        p.requires_grad = False # supposed to speed up training dramatically

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = make_training_args(str(modelsDir / "runs"))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=trainDs,
        eval_dataset=valDs, # monitoring val metrics each epoch
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=computeMetrics,
    )

    print("Starting fine-tuning ...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # saving training log + curves
    log_hist = pd.DataFrame(trainer.state.log_history)
    log_hist.to_csv(tablesDir / "distilbert_train_log.csv", index=False)
    
    # ensuring there is at least one validation loss/accuracy point
    # runs a single evaluation on the validation set and appends a row with epoch=1.0
    if "eval_loss" not in log_hist.columns or "eval_accuracy" not in log_hist.columns:
        eval_metrics_val = trainer.evaluate(eval_dataset=valDs)
        extra_row = {
            "epoch": 1.0,
            "eval_loss": eval_metrics_val.get("eval_loss"),
            "eval_accuracy": eval_metrics_val.get("eval_accuracy"),
        }
        log_hist = pd.concat([log_hist, pd.DataFrame([extra_row])], ignore_index=True)

    # show train loss, val loss, and val accuracy vs epoch
    fig = plt.figure(figsize=(6,4))
    if "loss" in log_hist:
        plt.plot(log_hist["epoch"], log_hist["loss"], marker="o", label="train loss")
    if "eval_loss" in log_hist:
        plt.plot(log_hist["epoch"], log_hist["eval_loss"], marker="o", label="val loss")
    if "eval_accuracy" in log_hist:
        plt.plot(log_hist["epoch"], log_hist["eval_accuracy"], marker="o", label="val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("DistilBERT: Training & Validation Metrics")
    fig.tight_layout()
    fig.savefig(figsDir / "distilbert_train_val_curves.png", dpi=150); plt.close(fig)
    
    # final test evaluation (one time)
    print("Evaluating on TEST set ...")
    t1 = time.time()
    evalOut = trainer.evaluate(eval_dataset=testDs)
    infer_time = time.time() - t1
    evalOut["train_wall_time_sec"] = float(train_time)
    evalOut["test_infer_time_sec"] = float(infer_time)

    with open(tablesDir / "distilbert_finetune_metrics.json", "w") as f:
        json.dump(evalOut, f, indent=2)

    # confusion matrix for finetuned model
    preds = trainer.predict(testDs)
    yPred = np.argmax(preds.predictions, axis=-1)
    yTrue = preds.label_ids
    saveConfusion(figsDir / "distilbert_finetune_confusion.png", yTrue, yPred, "DistilBERT (finetuned)")

    # saving model/tokenizer
    finalModelDir = modelsDir / "finetuned"
    model.save_pretrained(finalModelDir)
    tokenizer.save_pretrained(finalModelDir)

    print("\nSaved:")
    print(f"- Model -> {finalModelDir}")
    print(f"- Curves -> {figsDir / 'distilbert_train_val_curves.png'}")
    print(f"- Metrics -> {tablesDir / 'distilbert_finetune_metrics.json'}")
    print(f"- ConfMat -> {figsDir / 'distilbert_finetune_confusion.png'}")

if __name__ == "__main__":
    main()
