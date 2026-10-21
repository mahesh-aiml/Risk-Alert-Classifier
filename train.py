"""Risk Alert Classifier - full training pipeline (Parts B to G).
Run:  python train.py
Creates: model/*.pkl and output/* (tables, plots, report)."""
import warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, classification_report)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from utils import build_features, FEATURE_COLUMNS

SEED = 42
OUT = "output"
df = pd.read_csv("data/Risk_Alert_Classifier_Dataset_4600.csv")

# ---------------- Part B: data understanding & prep ----------------
y = df["risk_status"]
X = build_features(df)[FEATURE_COLUMNS]
missing = df.isna().sum()
missing = missing[missing > 0].rename("missing_count").to_frame()
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
missing.to_csv(f"{OUT}/01_missing_values.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print("Train/test class share:", y_train.mean().round(4), y_test.mean().round(4))

# KNN imputer (fit on train only -> no leakage) + scaler
prep = Pipeline([("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())])
Xtr = prep.fit_transform(X_train)
Xte = prep.transform(X_test)
print("Missing after imputation:", np.isnan(Xtr).sum())

# class distribution plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
y.value_counts().sort_index().plot(kind="bar", ax=ax[0], color=["#4C9F70", "#D9534F"])
ax[0].set_title("Target distribution (0=Low, 1=High risk)"); ax[0].set_xlabel("risk_status"); ax[0].set_ylabel("count")
ax[0].tick_params(axis="x", rotation=0)
missing["missing_count"].plot(kind="barh", ax=ax[1], color="#5B8DB8")
ax[1].set_title("Missing values per column")
plt.tight_layout(); plt.savefig(f"{OUT}/02_class_and_missing.png", dpi=130); plt.close()

# correlation heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(pd.DataFrame(Xtr, columns=FEATURE_COLUMNS).assign(risk=y_train.values).corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Feature correlation heatmap (train, imputed)"); plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_heatmap.png", dpi=130); plt.close()

# ---------------- helpers ----------------
results, probas = [], {}

def evaluate(name, model, Xa, ya, Xb, yb, group):
    p_te = model.predict_proba(Xb)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pred).ravel()
    row = dict(group=group, model=name,
               train_accuracy=accuracy_score(ya, model.predict(Xa)),
               test_accuracy=accuracy_score(yb, pred),
               precision=precision_score(yb, pred, zero_division=0),
               recall_minority=recall_score(yb, pred),
               f1=f1_score(yb, pred), auc_roc=roc_auc_score(yb, p_te),
               TN=tn, FP_typeI=fp, FN_typeII=fn, TP=tp)
    results.append(row); probas[name] = p_te
    return row

def cm_plot(model, Xb, yb, title, fname, thr=0.5):
    pred = (model.predict_proba(Xb)[:, 1] >= thr).astype(int)
    cm = confusion_matrix(yb, pred)
    plt.figure(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Low", "Pred High"], yticklabels=["Actual Low", "Actual High"])
    plt.title(title); plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=130); plt.close()

# ---------------- Part C: baseline LR ----------------
lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, y_train)
evaluate("Logistic Regression (baseline)", lr, Xtr, y_train, Xte, y_test, "C_Baseline")
cm_plot(lr, Xte, y_test, "Baseline Logistic Regression", "04_cm_baseline_lr.png")
rep = classification_report(y_test, lr.predict(Xte), target_names=["Low", "High"])
open(f"{OUT}/04_baseline_classification_report.txt", "w").write(rep)
print(rep)
coef = pd.Series(lr.coef_[0], index=FEATURE_COLUMNS).sort_values()
plt.figure(figsize=(7, 8)); coef.plot(kind="barh", color=np.where(coef > 0, "#D9534F", "#4C9F70"))
plt.title("Logistic Regression coefficients (+ increases risk)"); plt.tight_layout()
plt.savefig(f"{OUT}/05_lr_coefficients.png", dpi=130); plt.close()

# ---------------- Part D: imbalance handling ----------------
samplers = {"Under-Sampling": RandomUnderSampler(random_state=SEED),
            "Over-Sampling": RandomOverSampler(random_state=SEED),
            "SMOTE": SMOTE(random_state=SEED),
            "ADASYN": ADASYN(random_state=SEED)}
fitted_lr = {}
sizes = {"Original": dict(y_train.value_counts())}
for nm, s in samplers.items():
    Xr, yr = s.fit_resample(Xtr, y_train)
    sizes[nm] = dict(pd.Series(yr).value_counts())
    m = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xr, yr)
    fitted_lr[nm] = m
    evaluate(f"LR + {nm}", m, Xtr, y_train, Xte, y_test, "D_Imbalance")
json.dump({k: {int(a): int(b) for a, b in v.items()} for k, v in sizes.items()}, open(f"{OUT}/06_resampled_class_counts.json", "w"), indent=2)

# ---------------- Part E: trees ----------------
dt = DecisionTreeClassifier(random_state=SEED).fit(Xtr, y_train)
evaluate("Decision Tree (untuned)", dt, Xtr, y_train, Xte, y_test, "E_Trees")
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1).fit(Xtr, y_train)
evaluate("Random Forest (untuned)", rf, Xtr, y_train, Xte, y_test, "E_Trees")

# ---------------- Part F: tuning ----------------
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
dt_space = {"max_depth": [3, 4, 5, 6, 8, 10, None], "min_samples_split": [2, 5, 10, 20, 40],
            "min_samples_leaf": [1, 2, 5, 10, 20], "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced"]}
dt_rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=SEED), dt_space, n_iter=25, scoring="f1",
                           cv=cv, random_state=SEED, n_jobs=-1).fit(Xtr, y_train)
evaluate("Decision Tree (RandomSearch)", dt_rs.best_estimator_, Xtr, y_train, Xte, y_test, "F_Tuned")

rf_space = {"n_estimators": [100, 150], "max_depth": [4, 6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5], "class_weight": [None, "balanced", "balanced_subsample"]}
rf_rs = RandomizedSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=1), rf_space, n_iter=8, scoring="f1",
                           cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=1).fit(Xtr, y_train)
evaluate("Random Forest (RandomSearch)", rf_rs.best_estimator_, Xtr, y_train, Xte, y_test, "F_Tuned")

# Grid search fine-tuning around best RF
b = rf_rs.best_params_
def around(v, opts): return sorted(set(o for o in opts if o is not None and abs(opts.index(o) - opts.index(v)) <= 1)) if v in opts else [v]
grid = {"n_estimators": [b["n_estimators"], b["n_estimators"] + 50],
        "max_depth": [d for d in around(b["max_depth"], [4, 6, 8, 10, 12]) ] if b["max_depth"] is not None else [None],
        "min_samples_leaf": sorted(set([b["min_samples_leaf"], b["min_samples_leaf"] + 1, max(1, b["min_samples_leaf"] - 1)])),
        "max_features": [b["max_features"]], "min_samples_split": [b["min_samples_split"]],
        "class_weight": [b["class_weight"]]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=1), grid, scoring="f1", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1).fit(Xtr, y_train)
evaluate("Random Forest (GridSearch)", rf_gs.best_estimator_, Xtr, y_train, Xte, y_test, "F_Tuned")
json.dump({"DT_random": {k: (v if v is None or isinstance(v, str) else float(v) if isinstance(v, float) else int(v)) for k, v in dt_rs.best_params_.items()},
           "RF_random": {k: (v if v is None or isinstance(v, str) else float(v) if isinstance(v, float) else int(v)) for k, v in rf_rs.best_params_.items()},
           "RF_grid": {k: (v if v is None or isinstance(v, str) else float(v) if isinstance(v, float) else int(v)) for k, v in rf_gs.best_params_.items()},
           "cv_f1": {"DT_random": float(dt_rs.best_score_), "RF_random": float(rf_rs.best_score_), "RF_grid": float(rf_gs.best_score_)}},
          open(f"{OUT}/07_best_hyperparameters.json", "w"), indent=2)

# ---------------- Part G: ROC + final model selection ----------------
res = pd.DataFrame(results).round(4)
res.to_csv(f"{OUT}/08_all_model_metrics.csv", index=False)

plt.figure(figsize=(9, 7))
for nm, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{nm} (AUC={roc_auc_score(y_test, p):.3f})", lw=1.6)
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate (FPR)"); plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title("ROC curves - all models (test set)"); plt.legend(fontsize=7, loc="lower right"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/09_roc_all_models.png", dpi=130); plt.close()

# bar comparison
cmp_ = res.set_index("model")[["recall_minority", "f1", "auc_roc"]]
cmp_.plot(kind="bar", figsize=(12, 5)); plt.title("Recall (minority) / F1 / AUC by model"); plt.xticks(rotation=40, ha="right")
plt.tight_layout(); plt.savefig(f"{OUT}/10_metric_comparison.png", dpi=130); plt.close()

# overfitting plot (train vs test accuracy)
tt = res[res.group.isin(["E_Trees", "F_Tuned"])].set_index("model")[["train_accuracy", "test_accuracy"]]
tt.plot(kind="bar", figsize=(9, 5)); plt.title("Train vs test accuracy (overfitting check)"); plt.ylim(0.7, 1.02)
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(f"{OUT}/11_overfitting_check.png", dpi=130); plt.close()

# Business rule: minimise false negatives -> pick best AUC candidate among tuned/ensemble models,
# then choose decision threshold using TRAIN out-of-fold predictions (no test leakage) so that recall >= 0.98.
cands = {"Random Forest (RandomSearch)": rf_rs.best_estimator_, "Random Forest (GridSearch)": rf_gs.best_estimator_}
best_name = max(cands, key=lambda n: res.set_index("model").loc[n, "auc_roc"])
best_model = cands[best_name]
oof = cross_val_predict(best_model, Xtr, y_train, cv=cv, method="predict_proba")[:, 1]
thr_grid = np.arange(0.05, 0.96, 0.01)
ok = [t for t in thr_grid if recall_score(y_train, (oof >= t).astype(int)) >= 0.98]
THRESHOLD = float(min(0.5, max(ok))) if ok else 0.30   # highest threshold keeping OOF recall >= 98%, never above 0.5 (business: minimise false negatives)
print("Best:", best_name, "| threshold:", THRESHOLD)

pred = (probas[best_name] >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
final = dict(model=best_name, threshold=round(THRESHOLD, 2), accuracy=accuracy_score(y_test, pred),
             precision=precision_score(y_test, pred), recall=recall_score(y_test, pred), f1=f1_score(y_test, pred),
             auc_roc=roc_auc_score(y_test, probas[best_name]), TN=int(tn), FP_typeI=int(fp), FN_typeII=int(fn), TP=int(tp))
final = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()}
json.dump(final, open(f"{OUT}/12_final_model_metrics.json", "w"), indent=2)
cm_plot(type("W", (), {"predict_proba": lambda s, X: best_model.predict_proba(X)})(), Xte, y_test,
        f"Final model @ threshold {THRESHOLD:.2f}", "13_cm_final_model.png", thr=THRESHOLD)

imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
plt.figure(figsize=(7, 8)); imp.plot(kind="barh", color="#5B8DB8"); plt.title("Final model - feature importance")
plt.tight_layout(); plt.savefig(f"{OUT}/14_feature_importance.png", dpi=130); plt.close()
imp.sort_values(ascending=False).to_csv(f"{OUT}/14_feature_importance.csv", header=["importance"])

# ---------------- save model artefacts ----------------
full = Pipeline([("imputer", prep.named_steps["imputer"]), ("scaler", prep.named_steps["scaler"]), ("model", best_model)])
joblib.dump(full, "model/risk_model.pkl")
joblib.dump(dict(features=FEATURE_COLUMNS, threshold=THRESHOLD, model_name=best_name, metrics=final,
                 feature_importance=imp.sort_values(ascending=False).to_dict()), "model/model_meta.pkl")
joblib.dump(lr, "model/baseline_logreg.pkl")
print(res[["model", "test_accuracy", "precision", "recall_minority", "f1", "auc_roc", "FP_typeI", "FN_typeII"]].to_string(index=False))
print(final)
