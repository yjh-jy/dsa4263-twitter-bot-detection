import sys, os
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import shap
from visualize import evaluate_and_plot, shap_summary_for_model
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, PredefinedSplit
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
import pandas as pd

def box_m_test(X, y):
    """
    Box's M test for equality of covariance matrices across classes.
    Returns: dict with M, df, pval, details per class.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    p = X.shape[1]
    n_k = []
    S_k = []
    for c in classes:
        Xc = X[y == c]
        n_k.append(len(Xc))
        # sample covariance with (n-1) denom
        S_k.append(np.cov(Xc, rowvar=False, ddof=1))
    n_k = np.array(n_k)
    N = n_k.sum()
    # pooled covariance
    Sp = sum((n_k[i]-1) * S_k[i] for i in range(len(classes))) / (N - len(classes))
    # core statistic
    term = 0.0
    for i in range(len(classes)):
        term += (n_k[i]-1) * np.log(np.linalg.det(S_k[i]) + 1e-12)
    M = (N - len(classes)) * np.log(np.linalg.det(Sp) + 1e-12) - term

    # small-sample correction (as per Box, 1949)
    c = ( (2*p**2 + 3*p - 1) / (6*(p+1)*(len(classes)-1)) ) * \
        ( sum(1/(n_k[i]-1) for i in range(len(classes))) - 1/(N - len(classes)) )
    M_adj = M * (1 - c)
    df = (len(classes)-1) * p*(p+1)//2
    pval = 1 - chi2.cdf(M_adj, df)

    return {
        "M": float(M_adj),
        "df": int(df),
        "pval": float(pval),
        "classes": classes.tolist(),
        "n_per_class": n_k.tolist()
    }

# Import standard train

df_train = pd.read_csv(os.path.join(DATA_DIR, 'twitter_train_processed.csv'), index_col=0)
df_val = pd.read_csv(os.path.join(DATA_DIR, 'twitter_val_processed.csv'), index_col=0)
df_test = pd.read_csv(os.path.join(DATA_DIR, 'twitter_test_processed.csv'), index_col=0)

X_train = df_train.drop(columns=['account_type'])
y_train = df_train['account_type']

X_val = df_val.drop(columns=['account_type'])
y_val = df_val['account_type']

X_test = df_test.drop(columns=['account_type'])
y_test = df_test['account_type']

# Import SMOTECV
df_train_smote = pd.read_csv(os.path.join(DATA_DIR, 'twitter_train_processed_SMOTE.csv'), index_col=0)
X_train_smote = df_train_smote.drop(columns=['account_type'])
y_train_smote = df_train_smote['account_type']

# Import ADASYN
df_train_adasyn = pd.read_csv(os.path.join(DATA_DIR, 'twitter_train_processed_adasyn.csv'), index_col=0)
X_train_adasyn = df_train_adasyn.drop(columns=['account_type'])
y_train_adasyn = df_train_adasyn['account_type']

#resampling datasets
train_resamples = [(X_train,y_train), (X_train_smote,y_train_smote), (X_train_adasyn, y_train_adasyn)]
label = ['original', 'smote', 'adasyn']
# Cross-Validation Parameter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scoring Parameter
scoring = "roc_auc"

# Building The Pipelines and models
def make_pipe(estimator):
    return Pipeline([
        ('clf', estimator)
    ])

models_and_grids = {
    "LogReg": {
        "model": make_pipe(LogisticRegression(max_iter=3000, class_weight='balanced', solver='saga', n_jobs=None)),
        "param_grid": {
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.1, 1, 10, 100]
        }
    },
    "LDA": {
        # shrinkage works with solver='lsqr' or 'eigen'
        "model": make_pipe(LDA(solver='lsqr')),
        "param_grid": {
            "clf__shrinkage": [None, 'auto']  # None ~ no shrinkage; 'auto' ~ Ledoit-Wolf
        }
    },
    "QDA": {
        "model": make_pipe(QDA()),
        "param_grid": {
            "clf__reg_param": [0.0, 1e-3, 1e-2, 0.1, 0.5]
        }
    },
    "GaussianNB": {
        "model": make_pipe(GaussianNB()),
        "param_grid": {
            "clf__var_smoothing": np.logspace(-12, -7, 6)
        }
    },
    # "SVM": {
    #     "model": make_pipe(SVC()),
    #     "param_grid": {
    #         'clf__C': [0.1, 1, 1.0],
    #         'clf__kernel': ['linear', 'rbf'],
    #         'clf__gamma': ['scale', 'auto']
    #     }
    # }
}

# Main Code

def main():
    results_summary = []
    for i in range(len(train_resamples)):
        X_train, y_train = train_resamples[i][0], train_resamples[i][1]
        label_name = label[i]


        print(f"\n=== Covariance Diagnostics — {label_name} ===")
        bm = box_m_test(X_train, y_train)
        print(f"Box's M (adj): {bm['M']:.3f} | df={bm['df']} | p-value={bm['pval']:.4g} | n per class={bm['n_per_class']}")


        X_gs = pd.concat([X_train, X_val], axis=0)
        y_gs = pd.concat([y_train, y_val], axis=0).values

        test_fold = np.r_[
            -1 * np.ones(len(X_train), dtype=int),
             0 * np.ones(len(X_val),   dtype=int)
        ]
        ps = PredefinedSplit(test_fold=test_fold)

        for name, cfg in models_and_grids.items():
            print(f"\n>>> Tuning {name} ...")
            gs = GridSearchCV(
                estimator=cfg["model"],
                param_grid=cfg["param_grid"],
                scoring=scoring,
                cv=ps,
                n_jobs=-1,
                refit=True,
                verbose=0
            )
            gs.fit(X_gs, y_gs)

            print(f"{name}_{label_name} best params: {gs.best_params_}")
            print(f"{name}_{label_name} CV best {scoring}: {gs.best_score_:.4f}")

            # Evaluate on holdout
            auc_test = evaluate_and_plot(gs.best_estimator_, X_test, y_test, name, label_name, threshold=0.5)

            # Calculate SHAP
            shap_summary_for_model(gs.best_estimator_, X_train, X_test, name, label_name)

            results_summary.append({
                "Model": name,
                "CV_AUC": gs.best_score_,
                "Test_AUC": auc_test,
                "Best_Params": gs.best_params_
            })

    summary_df = pd.DataFrame(results_summary).sort_values("Test_AUC", ascending=False)
    print("\n=== Model Comparison (sorted by Test AUC) ===")
    print(summary_df.to_string(index=False))


main()