import os, json, joblib, numpy as np, scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# confusion matrix visualization, will be used for other models as well
def saveConfusion(figPath: Path, yTrue, yPred, title: str):
    cm = confusion_matrix(yTrue, yPred, labels=[0,1]) # 2x2 for true negatives (TN), false positives (FP), FN, TP, respectively 
    fig = plt.figure(figsize=(4.2,3.6)); ax = plt.gca()
    im = ax.imshow(cm); ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["neg","pos"]); ax.set_yticks([0,1]); ax.set_yticklabels(["neg","pos"]) # labels 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center") # formatting 
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(figPath, dpi=150); plt.close(fig) 

# calculating key classification performance metrics for logreg model, will be used for other models as well
def metricsDict(yTrue, yPred):
    acc = accuracy_score(yTrue, yPred) # how often predictions match true labels, acc(uracy) = 0.0-1.0
    precision, recall, f1, _ = precision_recall_fscore_support(yTrue, yPred, average="binary", pos_label=1) # focusing on positive 
    # class of reviews
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    # putting into dict so can be saved to json and compared to other models later

def main():
    processedDir = Path("ProjectB/data/processed")
    reportsDir = Path("ProjectB/reports")
    figsDir = Path(os.path.join(reportsDir, "figures"))
    tablesDir = Path(os.path.join(reportsDir, "tables"))
    modelsDir = Path("ProjectB/models/classical")
    for d in [figsDir, tablesDir, modelsDir]: 
        d.mkdir(parents=True, exist_ok=True)

    # loading TF-IDF artifacts from preprocess.py
    XTrain = sp.load_npz(processedDir / "X_train.npz")
    XTest = sp.load_npz(processedDir / "X_test.npz")
    yTrain = np.load(processedDir / "y_train.npy")
    yTest = np.load(processedDir / "y_test.npy")

    print("Training Logistic Regression ...")
    logReg = LogisticRegression(max_iter=1000, n_jobs=-1, solver="liblinear")
    logReg.fit(XTrain, yTrain)
    yPred = logReg.predict(XTest)

    mets = metricsDict(yTest, yPred)
    print("LogReg metrics:", mets)
    with open(tablesDir / "logreg_metrics.json", "w") as f: json.dump(mets, f, indent=2)
    with open(tablesDir / "logreg_classification_report.txt", "w") as f:
        f.write(classification_report(yTest, yPred, target_names=["negative","positive"]))
    saveConfusion(figsDir / "logreg_confusion.png", yTest, yPred, "LogReg Confusion Matrix")
    joblib.dump(logReg, modelsDir / "logreg.pkl")

    print("\nSaved:")
    print(f"- Model -> {modelsDir / 'logreg.pkl'}")
    print(f"- Figure -> {figsDir / 'logreg_confusion.png'}")
    print(f"- Metrics -> {tablesDir / 'logreg_metrics.json'}")

if __name__ == "__main__":
    main()
