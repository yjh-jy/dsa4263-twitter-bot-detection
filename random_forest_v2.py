import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib
import itertools
import hashlib

train_df = pd.read_csv('data/interim/twitter_train_processed.csv', index_col=0)
test_df = pd.read_csv('data/interim/twitter_test_processed.csv', index_col=0)

X_train = train_df.drop(columns=['account_type'])
y_train = train_df['account_type']

X_test_val = test_df.drop(columns=['account_type'])
y_test_val = test_df['account_type']

# 1:1 test-val split
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, stratify=y_test_val, random_state=42)

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

# Random Forest Evaluation
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
    plt.savefig(f"models/{label.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.title(f"ROC Curve (AUC = {auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()       
    plt.savefig(f"models/{label.lower().replace(' ', '_')}_roc_curve.png")
    plt.close()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y, y_prob)
    plt.plot(rec, prec, label=f"Classifier (AP = {pr_auc:.3f})")
    plt.title(f"Precision-Recall Curve")
    plt.xlabel("Recall (Positive label: 1)")
    plt.ylabel("Precision (Positive label: 1)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"models/{label.lower().replace(' ', '_')}_pr_curve.png")
    plt.close()    

    return auc, pr_auc, precision, recall, f1

best_val_score = 0
best_model_params = None
best_model = None

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", None],
    "bootstrap": [True, False]
}

# Convert label to a safe filename string
def safe_label(label): 
    hash_str = hashlib.md5(label.encode()).hexdigest()[:8]
    return f"{label.split()[0].lower()}_{hash_str}"

# Generate all combinations of hyperparameters
for combo in itertools.product(
    param_dist["n_estimators"],
    param_dist["max_depth"],
    param_dist["min_samples_split"],
    param_dist["min_samples_leaf"],
    param_dist["max_features"],
    param_dist["bootstrap"]
):

    params = {
        "n_estimators": combo[0],
        "max_depth": combo[1],
        "min_samples_split": combo[2],
        "min_samples_leaf": combo[3],
        "max_features": combo[4],
        "bootstrap": combo[5]
    }

    base_rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # True RFE
    rfe = RFE(estimator=base_rf, n_features_to_select=round(X_train.shape[1]*0.7), step=0.1)
    rfe.fit(X_train, y_train)
    
    X_train_rfe = X_train.loc[:, rfe.support_]
    X_val_rfe = X_val.loc[:, rfe.support_]

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train_rfe, y_train)

    safe_lbl = safe_lbl = safe_label(f"n{params['n_estimators']}_d{params['max_depth']}_s{params['min_samples_split']}_l{params['min_samples_leaf']}")
    _, pr_auc, _, _, _ = evaluate(model, X_val_rfe, y_val, label=f"Validation_{safe_lbl}")

    if pr_auc > best_val_score:
        best_val_score = pr_auc
        best_model_params = params
        best_model = model
        best_features = X_train.columns[rfe.support_]

print("\nBest hyperparameters:")
print(best_model_params)

# Refit best model on train+val sets
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

# Use only selected features
X_train_full_rfe = X_train_full[best_features]
X_test_rfe = X_test[best_features]

refitted_model = RandomForestClassifier(**best_model_params, random_state=42, n_jobs=-1)
refitted_model.fit(X_train_full_rfe, y_train_full)

# Save the refitted model
os.makedirs('models', exist_ok=True)
joblib.dump(refitted_model, 'models/best_rf_rfe.pkl')
joblib.dump(best_features, 'models/best_features_rfe.pkl')

# Evaluate on test set
print("\nEvaluation on test set:")
evaluate(refitted_model, X_test_rfe, y_test, label="Test")
