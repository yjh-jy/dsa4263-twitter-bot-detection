import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from scipy.stats import randint
import os
import joblib

# Load and prepare datasets
train_df = pd.read_csv('data/interim/twitter_train_processed.csv', index_col=0)
test_df = pd.read_csv('data/interim/twitter_test_processed.csv', index_col=0)

X = train_df.drop(columns=['account_type'])
y = train_df['account_type']

# 8:1:1 train-test-val split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

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

# Hyperparameter tuning
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Parameter grid to explore
param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": [None, 5, 10, 20, 30, 50],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

# Randomized search setup
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=30,                  
    scoring="roc_auc",          
    cv=3,                       
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit on training data
rf_random.fit(X_train, y_train)

print("\nBest Hyperparameters found:")
print(rf_random.best_params_)

# Retrieve and save best model
best_rf = rf_random.best_estimator_

os.makedirs('models', exist_ok=True)
joblib.dump(best_rf, 'models/best_rf.pkl')

# Evaluate on validation and test set
evaluate(best_rf, X_val, y_val, label="Validation")
evaluate(best_rf, X_test, y_test, label="Test")

# Predict and evaluate on actual test dataset
X_actual_test = test_df.drop(columns=['account_type'])
y_actual_test = test_df['account_type']

print("\nActual Test Dataset Evaluation")
evaluate(best_rf, X_actual_test, y_actual_test, label="Actual Test")
