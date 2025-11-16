import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
def main():
    # Location of outputs
    out_dir = "outputs"
    plot_dir = os.path.join(out_dir, "multimodal" , "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Models to evaluate
    models = ["text_only", "image_only", "concat", "gmu", "cross_attention"]

    # Store combined curve values
    roc_curves = {}
    pr_curves = {}

    # Loop through models
    for m in models:
        arr = np.load(os.path.join(out_dir, f"test_preds_{m}.npz"))
        y_true, y_prob, y_pred = arr["y_true"], arr["y_prob"], arr["y_pred"]

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Human (0)", "Bot (1)"],
                    yticklabels=["Human (0)", "Bot (1)"])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix — {m}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"cm_{m}.png"), dpi=200)
        plt.close()

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_curves[m] = (fpr, tpr, roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {m}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"roc_{m}.png"), dpi=200)
        plt.close()

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        pr_curves[m] = (rec, prec, ap)

        plt.figure()
        plt.plot(rec, prec, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve — {m}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"pr_{m}.png"), dpi=200)
        plt.close()

    # --- Combined ROC plot
    plt.figure(figsize=(6, 5))
    for m, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "roc_combined.png"), dpi=200)
    plt.close()

    # --- Combined PR plot
    plt.figure(figsize=(6, 5))
    for m, (rec, prec, ap) in pr_curves.items():
        plt.plot(rec, prec, label=f"{m} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pr_combined.png"), dpi=200)
    plt.close()

    print(f"All plots saved to {plot_dir}")
    
if __name__ == "__main__":
    main()