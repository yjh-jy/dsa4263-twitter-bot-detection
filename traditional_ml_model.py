import matplotlib.pyplot as plt
import numpy as np
import shap
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
        ('preprocess', pipe),  
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
    "SVM": {
        "model": make_pipe(SVC()),
        "param_grid": {
            'clf__C': [0.1, 1, 1.0],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale', 'auto']
        }
    }
}

# Display Results

def evaluate_and_plot(fitted_model, X_test, y_test, title, threshold=0.5,):

    if hasattr(fitted_model, "predict_proba"):
        y_prob = fitted_model.predict_proba(X_test)[:, 1]
    elif hasattr(fitted_model, "decision_function"):
        s = fitted_model.decision_function(X_test)
        y_prob = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        raise ValueError("Model lacks predict_proba/decision_function.")

    y_pred = (y_prob >= threshold).astype(int)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human (0)", "Bot (1)"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
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
    return auc


def shap_summary_for_model(fitted_model, X_train, X_test, model_name, max_bg=200, max_test=300, random_state=42):
    """
    SHAP on a fitted sklearn Pipeline:
      - Transforms X via pipeline's 'preprocess'
      - Explains the inner classifier ('clf') on numeric features
      - Works whether the clf exposes predict_proba or decision_function
    Produces a beeswarm + bar plot. Keeps runtime light via sampling.
    """
    # 1) Get steps
    if "preprocess" not in fitted_model.named_steps or "clf" not in fitted_model.named_steps:
        print(f"[SHAP] Skipping {model_name}: pipeline must have 'preprocess' and 'clf' steps.")
        return
    preprocessor = fitted_model.named_steps["preprocess"]
    clf = fitted_model.named_steps["clf"]

    # 2) Transform X to numeric (dense DataFrames with feature names)
    def _to_dense_df(X_raw):
        Xt = preprocessor.transform(X_raw)
        # Sparse -> dense if needed
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        # Try to get feature names; otherwise fall back to generic
        try:
            cols = preprocessor.get_feature_names_out()
        except Exception:
            cols = [f"f{i}" for i in range(np.asarray(Xt).shape[1])]
        return pd.DataFrame(Xt, columns=cols)

    X_train_t = _to_dense_df(X_train)
    X_test_t  = _to_dense_df(X_test)

    # 3) Light background + display slices (keep fast)
    bg = shap.sample(X_train_t, min(max_bg, len(X_train_t)), random_state=random_state)
    X_disp = shap.sample(X_test_t, min(max_test, len(X_test_t)), random_state=random_state)

    # 4) Prediction function on the *classifier* (post-preprocessing data)
    if hasattr(clf, "predict_proba"):
        pred_fn = lambda data: clf.predict_proba(data)  # -> (n, 2) for binary
        class_index = 1  # positive class
    elif hasattr(clf, "decision_function"):
        # decision_function returns (n,) or (n,1) — wrap to (n,1)
        pred_fn = lambda data: np.atleast_2d(clf.decision_function(data)).T
        class_index = 0
    else:
        print(f"[SHAP] Skipping {model_name}: classifier lacks predict_proba/decision_function.")
        return

    # 5) Build explainer and compute SHAP values
    # Model-agnostic Explainer handles any sklearn estimator here
    explainer = shap.Explainer(pred_fn, bg)
    sv = explainer(X_disp)

    # 6) Select the correct output (prob class=1 or single score)
    # sv.values shapes:
    #  - (n, n_features) for single output
    #  - (n, n_features, n_outputs) for multi-output (e.g., predict_proba)
    if getattr(sv.values, "ndim", 2) == 3:
        sv_pos = sv[:, :, class_index]
    else:
        sv_pos = sv

    # 7) Plots
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(sv_pos, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {model_name}" + (" (class=1)" if class_index == 1 else ""))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    shap.plots.bar(sv_pos, show=False, max_display=15)
    plt.title(f"SHAP Top Features — {model_name}" + (" (class=1)" if class_index == 1 else ""))
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
    auc_test = evaluate_and_plot(gs.best_estimator_, X_test, y_test, name, threshold=0.5)

    # Calculate SHAP
    shap_summary_for_model(gs.best_estimator_, X_train, X_test, name)

    results_summary.append({
        "Model": name,
        "CV_AUC": gs.best_score_,
        "Test_AUC": auc_test,
        "Best_Params": gs.best_params_
    })

summary_df = pd.DataFrame(results_summary).sort_values("Test_AUC", ascending=False)
print("\n=== Model Comparison (sorted by Test AUC) ===")
print(summary_df.to_string(index=False))

