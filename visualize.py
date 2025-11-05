import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score, f1_score, accuracy_score
)
import shap



# Plot Confusion Matrix and AUC

def evaluate_and_plot(fitted_model, X_test, y_test, title, label, threshold=0.5,):

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
    ax[0].set_title(f"Confusion Matrix {title}_{label}")

    #ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax[1])
    ax[1].set_title(f"ROC Curve (AUC = {auc:.3f}) {title}_{label}")

    # Precicision Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax[2])
    ax[2].set_title(f"Precision–Recall Curve {title}_{label}")

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

# Plot SHAP
def shap_summary_for_model(fitted_model, X_train, X_test, model_name, max_bg=200, max_test=300, random_state=42):
    """
    SHAP on a fitted sklearn Pipeline:
      - Transforms X via pipeline's 'preprocess'
      - Explains the inner classifier ('clf') on numeric features
      - Works whether the clf exposes predict_proba or decision_function
    Produces a beeswarm + bar plot. Keeps runtime light via sampling.
    """
    # 1) Get steps
    if "clf" not in fitted_model.named_steps:
        print(f"[SHAP] Skipping {model_name}: pipeline must have 'preprocess' and 'clf' steps.")
        return
    clf = fitted_model.named_steps["clf"]


    # 3) Light background + display slices (keep fast)
    bg = shap.sample(X_train, min(max_bg, len(X_train)), random_state=random_state)
    X_disp = shap.sample(X_test, min(max_test, len(X_test)), random_state=random_state)

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