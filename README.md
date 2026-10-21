# 🚨 Risk Alert Classifier

Early-warning classifier for a digital bank: flags high-risk customers (`risk_status = 1`).

## Folder structure
| Path | Contents |
|---|---|
| `app.py` | Streamlit app (single prediction, batch CSV prediction, model report) |
| `train.py` | Full pipeline: split → KNN imputer → LR baseline → under/over/SMOTE/ADASYN → DT/RF → RandomSearch/GridSearch → ROC → final model |
| `utils.py` | Shared feature engineering (used by training and app) |
| `model/risk_model.pkl` | Final pipeline (KNNImputer + StandardScaler + tuned Random Forest) |
| `model/model_meta.pkl` | Threshold, metrics, feature importance |
| `model/baseline_logreg.pkl` | Baseline Logistic Regression |
| `output/` | Metric tables (CSV/JSON), all plots (PNG), training log |
| `data/` | Dataset |

## Run
```bash
pip install -r requirements.txt
python train.py          # optional - regenerates model/ and output/
streamlit run app.py
```

## Method
- Target: `risk_status`. `customer_id` dropped. `last_transaction_date` → `days_since_last_txn`; added `debt_to_income`; categoricals one-hot encoded (missing category = all zeros).
- 80/20 stratified split (≈12.1% high-risk in both). KNN Imputer (k=5) and scaler are fit on the **train set only**.
- Resampling is applied to the training set only; the test set is never resampled.
- RandomizedSearchCV (Decision Tree, Random Forest) then GridSearchCV around the best RF, scoring = F1.
- Final threshold is chosen on **train out-of-fold** predictions (recall ≥ 98%, capped at 0.5), because missing a risky customer (false negative) is the costly error.

## Results (test set, 920 rows)
See `output/08_all_model_metrics.csv` for all models and `output/12_final_model_metrics.json` for the final one.

Final model: tuned Random Forest at threshold 0.35 → Recall 1.00, Precision 0.94, F1 0.97, AUC 1.00, 0 false negatives, 7 false positives.

### Important note
This dataset is almost perfectly separable: even the baseline Logistic Regression scores ~100% on the test set, so imbalance handling and tuning add little here. This points to a synthetic dataset whose label follows a simple rule from features such as late-payment days, missed payments, and credit score. Real bank data would be much harder, and these scores should not be expected to carry over.

## Business interpretation
- **False Negative (Type-II):** a risky customer is missed, leading to defaults or fraud losses. This is the costly error, so the threshold favours recall.
- **False Positive (Type-I):** a safe customer is flagged, leading to extra manual review or customer friction. Cheaper, so a few are acceptable.
