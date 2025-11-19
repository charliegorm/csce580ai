import json
from pathlib import Path
import pandas as pd
import os

def loadj(p: Path):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("WARNING: missing", p)
        return {}
    except json.JSONDecodeError:
        print("WARNING: bad JSON in", p)
        return {}

def main():
    tables = Path("ProjectB/reports/tables")

    rows = []

    # logreg
    logreg = loadj(Path(os.path.join(tables, "logreg_metrics.json")))
    rows.append({
        "model": "Logistic Regression",
        "accuracy": logreg.get("accuracy"),
        "precision": logreg.get("precision"),
        "recall": logreg.get("recall"),
        "f1": logreg.get("f1"),
        "train_wall_time_sec": None,  # not measured
        "test_infer_time_sec": None   # optional to measure later
    })

    # DistilBERT finetuned
    db_ft = loadj(Path(os.path.join(tables, "distilbert_finetune_metrics.json")))
    rows.append({
        "model": "DistilBERT (finetuned)",
        "accuracy": db_ft.get("eval_accuracy"),
        "precision": db_ft.get("eval_precision"),
        "recall": db_ft.get("eval_recall"),
        "f1": db_ft.get("eval_f1"),
        "train_wall_time_sec": db_ft.get("train_wall_time_sec"),
        "test_infer_time_sec": db_ft.get("test_infer_time_sec"),
    })

    # DistilBERT base
    db_base = loadj(Path(os.path.join(tables, "distilbert_base_metrics.json")))
    rows.append({
        "model": "DistilBERT (base)",
        "accuracy": db_base.get("accuracy"),
        "precision": db_base.get("precision"),
        "recall": db_base.get("recall"),
        "f1": db_base.get("f1"),
        "train_wall_time_sec": None,  # no training
        "test_infer_time_sec": db_base.get("test_infer_time_sec"),
    })

    # GPT2 
    gpt2 = loadj(Path(os.path.join(tables, "gpt2_metrics.json")))
    rows.append({
        "model": "GPT-2",
        "accuracy": gpt2.get("accuracy"),
        "precision": gpt2.get("precision"),
        "recall": gpt2.get("recall"),
        "f1": gpt2.get("f1"),
        "train_wall_time_sec": None,  # pre-trained, we didn't train it
        "test_infer_time_sec": gpt2.get("test_infer_time_sec"),
    })

    df = pd.DataFrame(rows)

    # column order
    cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "train_wall_time_sec",
        "test_infer_time_sec",
    ]
    df = df[cols]

    out_csv = tables / "comparison_table.csv"
    out_json = tables / "comparison_table.json"

    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)

    print("Wrote: ", out_csv)
    print("Wrote: ", out_json)

if __name__ == "__main__":
    main()
