import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from preprocessing import pipe
import pandas as pd


df = pd.read_csv('data/raw/twitter_human_bots_dataset.csv', index_col=0)
X = df.drop(columns=['account_type', 'id'])
y = df['account_type'].map({'bot':1, 'human':0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = "roc_auc"

# Building The Pipelines

def make_pipe(estimator):
    return Pipeline([
        ('preprocess', pipe),   # your FE + ColumnTransformer
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
    # "SVM_RBF": {
    #     "model": make_pipe(SVC(kernel='rbf', probability=True, class_weight='balanced')),
    #     "param_grid": {
    #         "clf__C": [0.1, 1, 10],
    #         "clf__gamma": ["scale", 0.01, 0.1, 1.0]
    #     }
    # },
    # "SVM_Linear": {
    #     # using SVC with linear kernel to keep probability=True support
    #     "model": make_pipe(SVC(kernel='linear', probability=False, class_weight='balanced')),
    #     "param_grid": {
    #         "clf__C": [0.1, 1, 10, 100]
    #     }
    # },
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
    }
}



# Display Results

def evaluate_and_plot(fitted_model, X_test, y_test, title, threshold=0.5,):
    fitted_model.fit(X_train, y_train)

    if hasattr(fitted_model, "predict_proba"):
        y_prob = fitted_model.predict_proba(X_test)[:, 1]
    elif hasattr(fitted_model, "decision_function"):
        # score to [0,1] for plotting/thresholding convenience
        s = fitted_model.decision_function(X_test)
        y_prob = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        raise ValueError("Model lacks predict_proba/decision_function.")

    y_pred = (y_prob >= threshold).astype(int)

    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human (0)", "Bot (1)"])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    disp.plot(ax=ax[0], cmap="Blues", colorbar=False)
    ax[0].set_title("Confusion Matrix")

    #ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax[1])
    ax[1].set_title(f"ROC Curve (AUC = {auc:.3f})")

    # Precicision Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax[2])
    ax[2].set_title("Precision–Recall Curve")

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    plt.tight_layout()
    plt.show()




results_summary = []

for name, cfg in models_and_grids.items():
    print(f"\n>>> Tuning {name} ...")
    gs = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["param_grid"],
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    gs.fit(X_train, y_train)

    print(f"{name} best params: {gs.best_params_}")
    print(f"{name} CV best {scoring}: {gs.best_score_:.4f}")

    # Evaluate on holdout
    evaluate_and_plot(gs.best_estimator_, X_test, y_test, name, threshold=0.5)

    # Store summary for quick comparison
    # (Compute AUC on test quickly for a table)
    if hasattr(gs.best_estimator_, "predict_proba"):
        y_prob_test = gs.best_estimator_.predict_proba(X_test)[:, 1]
    else:
        s = gs.best_estimator_.decision_function(X_test)
        y_prob_test = (s - s.min()) / (s.max() - s.min() + 1e-12)
    auc_test = roc_auc_score(y_test, y_prob_test)

    results_summary.append({
        "Model": name,
        "CV_AUC": gs.best_score_,
        "Test_AUC": auc_test,
        "Best_Params": gs.best_params_
    })

summary_df = pd.DataFrame(results_summary).sort_values("Test_AUC", ascending=False)
print("\n=== Model Comparison (sorted by Test AUC) ===")
print(summary_df.to_string(index=False))

