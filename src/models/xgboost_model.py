import numpy as np
import pandas as pd
import optuna
import joblib
import json
import sys

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, precision_recall_curve
from src.viz.visualize_utility import evaluate_and_plot, shap_summary_for_model_xgboost

optuna.logging.set_verbosity(optuna.logging.ERROR)

def artifact_paths(name):
        # Return file paths for artifacts for a given dataset name
        return {
            "model": Path(f"models/xgboost/final_model_{name}.joblib"),
            "features": Path(f"models/xgboost/best_features_{name}.joblib"),
            "meta": Path(f"models/xgboost/study_meta_{name}.json"),
            "study":  Path(f"models/xgboost/optuna_study_{name}.joblib")
        }

def load_artifacts_joblib(name):
    p = artifact_paths(name)
    if p["model"].exists() and p["features"].exists() and p["meta"].exists():
        model = joblib.load(p["model"])
        features = joblib.load(p["features"])
        with open(p["meta"], "r") as f:
            meta = json.load(f)
        return model, features, meta
    return None, None, None

def save_artifacts_joblib(name, study, final_model, best_cols):
    p = artifact_paths(name)
    joblib.dump(final_model, p["model"])
    joblib.dump(best_cols, p["features"])
    meta = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study_name": study.study_name
    }
    with open(p["meta"], "w") as f:
        json.dump(meta, f, indent=2)

def make_objective(x, y, seed, min_precision=0.7, n_splits=5):
    def objective(trial):
        pos = int(y.sum())
        neg = len(y) - pos
        default_spw = max(1.0, neg / max(1.0, pos))

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=False),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, default_spw * 3.0, log=True),
        }

        feature_frac = trial.suggest_float("feature_fraction", 0.60, 0.95)
        n_features_to_select = max(1, int(round(x.shape[1] * feature_frac)))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        recalls = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            X_tr, X_val_fold = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # RFE estimator 
            rfe_estimator = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                **params,
            )

            rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select, step=0.1)
            rfe.fit(X_tr, y_tr)

            selected_cols = X_tr.columns[rfe.support_]

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
                **params,
            )

            model.fit(X_tr.loc[:, selected_cols], y_tr)

            y_val_probs = model.predict_proba(X_val_fold.loc[:, selected_cols])[:, 1]
            precisions, recalls_pr, threshold = precision_recall_curve(y_val_fold, y_val_probs)

            valid_mask = precisions[:-1] >= min_precision
            valid_idxs = np.where(valid_mask)[0]

            best_idx = valid_idxs[np.argmax(recalls_pr[valid_idxs])]
            chosen_recall = float(recalls_pr[best_idx])

            recalls.append(chosen_recall)

        return float(np.mean(recalls)) if len(recalls) > 0 else 0.0

    return objective

def main():
    train_df = pd.read_csv("data/cleaned/twitter_train_processed.csv")
    test_df = pd.read_csv("data/cleaned/twitter_test_processed.csv")
    train_adasyn_df = pd.read_csv("data/cleaned/twitter_train_processed_adasyn.csv")
    train_smote_df = pd.read_csv("data/cleaned/twitter_train_processed_smote.csv")

    seed = 42

    # 1:1 test-val split
    X_test_val = test_df.drop(columns=["account_type"])
    y_test_val = test_df["account_type"]
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_val, y_test_val, test_size=0.5, stratify=y_test_val, random_state=seed
    )

    # Training set
    X_train = train_df.drop(columns=['account_type'])
    X_train_adasyn = train_adasyn_df.drop(columns=['account_type'])
    X_train_smote = train_smote_df.drop(columns=['account_type'])

    y_train = train_df['account_type']
    y_train_adasyn = train_adasyn_df['account_type']
    y_train_smote = train_smote_df['account_type']

    # Combine train + val for hyperparameter tuning
    X_tune, y_tune = pd.concat([X_train, X_val], axis=0), pd.concat([y_train, y_val], axis=0)
    X_tune_adasyn, y_tune_adasyn = pd.concat([X_train_adasyn, X_val], axis=0), pd.concat([y_train_adasyn, y_val], axis=0)
    X_tune_smote, y_tune_smote = pd.concat([X_train_smote, X_val], axis=0), pd.concat([y_train_smote, y_val], axis=0)

    force_retrain = False # set True to ignore saved artifacts and re-run tuning/refit

    studies = {}
    models = {}
    best_features = {}
    metrics = {}

    datasets = {
        "no_resampling": (X_tune, y_tune),
        "adasyn": (X_tune_adasyn, y_tune_adasyn),
        "smote": (X_tune_smote, y_tune_smote),
    }

    for name, (x, y) in datasets.items():
        model_saved, cols_saved, meta_saved = load_artifacts_joblib(name)
        if model_saved is not None and not force_retrain:
            print(f"\nLoaded saved artifacts for {name}: best_value={meta_saved.get('best_value')}")
            studies[name] = None 
            models[name] = model_saved
            best_features[name] = cols_saved
            metrics[name] = evaluate_and_plot(model_saved, X_test.loc[:, cols_saved], y_test, model_name='XGBoost', dataset_name=name)

            continue  # skip tuning/refit for this dataset

        study = optuna.create_study(direction="maximize", study_name=f"xgb_{name}")
        obj = make_objective(x, y, seed, min_precision=0.7, n_splits=5)
        study.optimize(obj, n_trials=40, show_progress_bar=True)
        studies[name] = study
        print(f"\nFinished study for {name}: best_value={study.best_value}")

        # Extract best parameters
        best_params = {k: v for k, v in study.best_params.items() if k != "feature_fraction"}
        best_feature_frac = study.best_params.get("feature_fraction", 0.7)
        n_features_to_select_full = max(1, int(round(x.shape[1] * best_feature_frac)))

        # Fit RFE on full tuning set using best params
        rfe_estimator_full = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            **best_params,
        )

        rfe_full = RFE(estimator=rfe_estimator_full, n_features_to_select=n_features_to_select_full, step=0.1)
        rfe_full.fit(x, y)
        best_cols = x.columns[rfe_full.support_].tolist()
        best_features[name] = best_cols
        print(f"Refit {len(best_cols)} selected features for {name}")

        # Train final model on these features
        final_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            **best_params,
        )

        final_model.fit(x.loc[:, best_cols], y)
        models[name] = final_model

        # Evaluate on test set
        metrics[name] = evaluate_and_plot(model_saved, X_test.loc[:, cols_saved], y_test, model_name='XGBoost', dataset_name=name)

        # Save artifacts (model, features, meta)
        save_artifacts_joblib(name, study, final_model, best_cols)

    # Compare test recall and PR-AUC
    results = {}
    print("")
    for name, model in models.items():
        cols = best_features[name]
        X_test_sub = X_test.loc[:, cols]

        # Get positive-class probabilities
        y_prob = model.predict_proba(X_test_sub)[:, 1]

        # Convert probabilities to binary predictions
        y_pred = (y_prob >= 0.5).astype(int)

        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, y_prob)

        results[name] = {"recall": rec, "pr_auc": pr_auc, "y_prob": y_prob}
        print(f"{name}: recall={rec:.4f}, PR-AUC={pr_auc:.4f}, precision={prec:.4f}, n_features={len(cols)}")

    # Create sorted list by (recall, pr_auc) descending
    rows = [(n, v["recall"], v["pr_auc"]) for n, v in results.items()]
    rows_sorted = sorted(rows, key=lambda x: (x[1], x[2]), reverse=True)
    best_name, best_recall, best_pr_auc = rows_sorted[0]
    print(f"\nSelected best dataset: {best_name} (recall={best_recall:.4f}, PR-AUC={best_pr_auc:.4f}, precision={prec:.4f})")

    best_model = models[best_name]
    best_cols = best_features[best_name]
    X_test_sub = X_test.loc[:, best_cols]

    shap_summary_for_model_xgboost(best_model, X_test_sub, y_test)
    
if __name__ == "__main__":
    main()