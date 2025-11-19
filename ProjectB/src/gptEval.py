import json, time, torch, numpy as np, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import os

def ensureDir(p: Path): p.mkdir(parents=True, exist_ok=True)

# device selection, gpt cant handle everything
device = "mps" if torch.backends.mps.is_available() else "cpu"

# respect model max length & set pad token
def load_gpt2(name="openai-community/gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    # GPT-2 has no pad token by default, using EOS to avoid warnings errors when padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id
    model.to(device)
    model.eval()
    return tok, model

def score_sentiment(model, tok, text: str):
    prompt = f"Review:\n{text}\n\nSentiment:"
    # truncation to model max, also sending tensors to device for run time
    inputs = tok(prompt,
                 return_tensors="pt",
                 truncation=True,
                 max_length=min(tok.model_max_length, 1024),
                 padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :]  # next-token logits

    # gpt2 usually has single tokens for leading-space words
    tok_pos = tok.encode(" positive", add_special_tokens=False)
    tok_neg = tok.encode(" negative", add_special_tokens=False)
    pos_logit = logits[tok_pos[0]] if tok_pos else -1e9
    neg_logit = logits[tok_neg[0]] if tok_neg else -1e9
    return 1 if pos_logit >= neg_logit else 0

def main():
    interimDir = Path("ProjectB/data/interim")
    reportsDir = Path("ProjectB/reports")
    figsDir = Path(os.path.join(reportsDir, "figures"))
    tablesDir = Path(os.path.join(reportsDir, "tables"))
    for d in [figsDir, tablesDir]:
        ensureDir(Path(d))

    print("Loading test.csv ...")
    testDf = pd.read_csv(interimDir / "test.csv")  # columns: review,label

    print("Loading GPT-2 ...")
    tok, model = load_gpt2("openai-community/gpt2")  # sets pad token & device

    preds = []
    t0 = time.time()
    for text in testDf["review"]:
        preds.append(score_sentiment(model, tok, str(text)))
    infer_time = time.time() - t0

    yTrue = testDf["label"].to_numpy()
    yPred = np.array(preds, dtype=int)

    acc = accuracy_score(yTrue, yPred)
    pr, rc, f1, _ = precision_recall_fscore_support(yTrue, yPred, average="binary", pos_label=1)
    out = {"accuracy": float(acc), "precision": float(pr), "recall": float(rc), "f1": float(f1), "test_infer_time_sec": float(infer_time)}
    with open(tablesDir / "gpt2_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    cm = confusion_matrix(yTrue, yPred, labels=[0,1])
    fig = plt.figure(figsize=(4.2,3.6)); ax = plt.gca()
    im = ax.imshow(cm); ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["neg","pos"]); ax.set_yticklabels(["neg","pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("GPT-2 (zero-shot) Confusion"); fig.tight_layout()
    fig.savefig(figsDir / "gpt2_confusion.png", dpi=150); plt.close(fig)

    print("Saved GPT-2 metrics/confusion.")

if __name__ == "__main__":
    main()
