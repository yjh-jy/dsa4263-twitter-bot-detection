import pandas as pd
from sklearn.model_selection import train_test_split #, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
# from scipy.stats import randint
import os
import joblib
import itertools

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

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    _, pr_auc, _, _, _ = evaluate(model, X_val, y_val, label=f"Validation (params={params})")
    
    if pr_auc > best_val_score:  
        best_val_score = pr_auc
        best_model_params = params
        best_model = model

print("\nBest hyperparameters:")
print(best_model_params)

# Refit best model on train+val sets
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

refitted_model = RandomForestClassifier(**best_model_params, random_state=42, n_jobs=-1)
refitted_model.fit(X_train_full, y_train_full)

# Save the refitted model
os.makedirs('models', exist_ok=True)
joblib.dump(refitted_model, 'models/best_rf.pkl')

# Evaluate on test set
print("\nEvaluation on test set:")
evaluate(refitted_model, X_test, y_test, label="Test")
