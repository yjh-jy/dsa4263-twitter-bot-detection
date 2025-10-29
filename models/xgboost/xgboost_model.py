import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

os.makedirs('models/xgboost', exist_ok=True)
root = os.environ.get("PROJECT_ROOT", os.getcwd())
train_df = pd.read_csv(os.path.join(root, "data", "interim", "twitter_train_processed.csv"))
test_df = pd.read_csv(os.path.join(root, "data", "interim", "twitter_test_processed.csv"))
seed = 42

# 1:1 test-val split
X_test_val = test_df.drop(columns=["account_type"])
y_test_val = test_df["account_type"]
X_val, X_test, y_val, y_test = train_test_split(
    X_test_val, y_test_val, test_size=0.5, stratify=y_test_val, random_state=seed
)

# Training set
X_train = train_df.drop(columns=["account_type"])
y_train = train_df["account_type"]

# Tuning set
X_tune = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_tune = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

def evaluate(model, X, y, label="Validation"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n{label} Set Metrics")
    print(f"AUC: {auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=["Human (0)", "Bot (1)"], cmap="Blues", colorbar=False)
    disp.ax_.set_title("Confusion Matrix")
    disp.ax_.set_xlabel("Predicted label")
    disp.ax_.set_ylabel("True label")
    plt.tight_layout()    
    plt.savefig(f"models/xgboost/{label.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.title(f"ROC Curve (AUC = {auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()       
    plt.savefig(f"models/xgboost/{label.lower().replace(' ', '_')}_roc_curve.png")
    plt.close()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y, y_prob)
    plt.plot(rec, prec, label=f"Classifier (AP = {pr_auc:.3f})")
    plt.title(f"Precision-Recall Curve")
    plt.xlabel("Recall (Positive label: 1)")
    plt.ylabel("Precision (Positive label: 1)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"models/xgboost/{label.lower().replace(' ', '_')}_pr_curve.png")
    plt.close()    

    return auc, pr_auc, precision, recall, f1

search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=seed, n_jobs=-1, verbosity=0),
    param_distributions={
        "n_estimators": randint(200, 800),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 7),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4)
    },
    n_iter=25,
    scoring="roc_auc",
    n_jobs=-1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    verbose=1,
    random_state=seed
)

# Tuning
search.fit(X_tune, y_tune)
best_model = search.best_estimator_
print("Best hyperparameters:")
for param, value in search.best_params_.items():
    print(f"{param}: {value}")

# Evaluate on test set
metrics = evaluate(best_model, X_test, y_test)
print("\nEvaluation metrics:")
print(metrics)

# SHAP
explainer = shap.Explainer(best_model.predict_proba, X_test)
shap_values = explainer(X_test)
vals = np.asarray(shap_values.values)[:, :, 1]
global_importance = np.abs(vals).mean(axis=0)
shap_df = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": global_importance}).sort_values(by="mean_abs_shap", ascending=False)
top_features = shap_df.head(10)
print("\nTop 10 global features:")
print(top_features)

# Plot global features
plt.figure(figsize=(8, 6))
bars = plt.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1])
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1e-6, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center")
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP Top 10 Features")
plt.tight_layout()
plt.show()

# Subset SHAP for bots
target = 1
y_test_arr = np.asarray(y_test).ravel()
mask = (y_test_arr == target)

# Extract class-specific SHAP
vals_local = np.asarray(shap_values.values)
shap_vals_subset = vals_local[mask, :, 1]
mean_abs_subset = np.abs(shap_vals_subset).mean(axis=0)
X_test_subset = X_test.iloc[mask].reset_index(drop=True)
feature_df_subset = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap_subset": mean_abs_subset}).sort_values("mean_abs_shap_subset", ascending=False)
top_subset = feature_df_subset.head(10)
print("\nTop 10 features for bots:")
print(top_subset)

# Beeswarm
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_vals_subset, X_test_subset, plot_type="dot", max_display=10, show=False)
plt.title(f"SHAP Beeswarm (bots)")
plt.tight_layout()
plt.show()

# Bar plot
plt.figure(figsize=(8, 6))
bars = plt.barh(top_subset["feature"][::-1], top_subset["mean_abs_shap_subset"][::-1])
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1e-6, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center")
plt.xlabel("Mean |SHAP value| (subset)")
plt.title(f"Top 10 features by mean(|SHAP|) for bots")
plt.tight_layout()
plt.show()