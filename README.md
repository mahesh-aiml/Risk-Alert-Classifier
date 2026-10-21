# 🛡️ Risk Alert Classifier
 
A machine learning web app that estimates the probability of a customer being **High Risk** based on their financial profile, repayment behaviour and account activity. Built with **Python, scikit-learn and Streamlit**.
 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
 
---
 
## 📌 Table of Contents
 
- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Input Features](#-input-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How the Input Schema Works](#-how-the-input-schema-works)
- [Troubleshooting](#-troubleshooting)
- [Disclaimer](#-disclaimer)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
---
 
## 🔍 Overview
 
Financial institutions need to flag risky customers early. This project wraps a trained classification model in a simple Streamlit interface: enter a customer's details, click **Predict Risk**, and get:
 
- the **High Risk** and **Low Risk** probabilities
- the final predicted risk status
- a table of the submitted data for review
The form is generated dynamically from a JSON schema, so the interface adapts automatically when features change.
 
## ✨ Features
 
- 🎛️ **Dynamic input form** generated from `input_schema.json`
- 📊 **Probability output** for both High Risk and Low Risk classes
- 🔢 **Smart numeric inputs**: whole numbers for counts, decimals for ratios
- 🔁 **Automatic feature ordering** that matches the model's expected columns
- 🧩 **Flexible schema loader** supporting three JSON formats
- 🛑 **Clear error messages** for missing files, bad schema or feature mismatch
- ⚡ **Cached model loading** for fast predictions
## 🎬 Demo
 
> Add a screenshot or GIF of your app here.
 
```
![App Screenshot](assets/screenshot.png)
```
 
## 🧾 Input Features
 
| Feature | Type | Range / Options |
|---|---|---|
| `age` | Numeric | 18 – 75 |
| `gender` | Categorical | Female, Male, Other |
| `region` | Categorical | Central, East, North, South, West |
| `employment_type` | Categorical | Retired, Salaried, Self-Employed, Student, Unemployed |
| `annual_income_inr` | Numeric | 15,000 – 163,002 |
| `credit_score` | Numeric | 440 – 850 |
| `credit_utilization_ratio` | Numeric | 0.002 – 0.978 |
| `missed_payments_12m` | Numeric | 0 – 10 |
| `avg_late_payment_days` | Numeric | 0.1 – 47.1 |
| `monthly_transaction_count` | Numeric | 5 – 153 |
| `monthly_spend_inr` | Numeric | 3,769 – 87,389 |
| `cash_advance_count_6m` | Numeric | 0 – 7 |
| `complaints_last_6m` | Numeric | 0 – 7 |
| `failed_login_attempts_3m` | Numeric | 0 – 12 |
| `account_tenure_months` | Numeric | 2 – 180 |
| `debt_balance_inr` | Numeric | 3,791 – 213,601 |
| `day_since_last_transaction` | Numeric | 0 – 89 |
 
## 📁 Project Structure
 
```
Project_3/
├── app.py                      # Streamlit application
├── model/
│   └── model.joblib            # Trained scikit-learn model
├── outputs/
│   └── input_schema.json       # Feature definitions used to build the form
├── requirements.txt            # Python dependencies
└── README.md
```
 
## ⚙️ Installation
 
**1. Clone the repository**
 
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```
 
**2. Create a virtual environment (recommended)**
 
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS / Linux
source venv/bin/activate
```
 
**3. Install dependencies**
 
```bash
pip install -r requirements.txt
```
 
Example `requirements.txt`:
 
```
streamlit
pandas
joblib
scikit-learn
```
 
> ⚠️ Install the **same scikit-learn version** you used to train the model, otherwise loading `model.joblib` may fail or give warnings.
 
## 🚀 Usage
 
```bash
streamlit run app.py
```
 
The app opens at `http://localhost:8501`.
 
1. Fill in the customer information.
2. Click **Predict Risk**.
3. Review the probabilities and the predicted status.
## 🧩 How the Input Schema Works
 
The form is built from `outputs/input_schema.json`. Each feature needs a `type`:
 
```json
{
  "age": {
    "type": "numeric",
    "default": 36.0,
    "min": 18.0,
    "max": 75.0
  },
  "gender": {
    "type": "categorical",
    "default": "Male",
    "choices": ["Female", "Male", "Other"]
  }
}
```
 
| Key | Used for | Description |
|---|---|---|
| `type` | All | `numeric` or `categorical` |
| `default` | All | Pre-selected value |
| `min`, `max` | Numeric | Allowed range |
| `choices` | Categorical | Dropdown options |
 
The loader also accepts a dictionary with a `"features"` list, or a plain list of feature dictionaries (each with a `name`).
 
**Important:** feature names in the schema must exactly match the column names the model was trained on.
 
## 🛠️ Troubleshooting
 
| Problem | Likely Cause | Fix |
|---|---|---|
| `KeyError: 0` | Old `app.py` using `config[0]` on a dict schema | Use the latest `app.py` and restart Streamlit |
| `Model file not found` | Wrong folder structure | Keep `model/model.joblib` next to `app.py` |
| `Schema file not found` | Schema in wrong location | Keep `outputs/input_schema.json` next to `app.py` |
| Feature names do not match | Schema key differs from training column | Rename the key in the JSON (for example `day_since_last_transaction`) |
| `Cannot identify the High Risk class` | Target encoded with unexpected labels | Add your label to `HIGH_RISK_LABELS` in `app.py` |
| Old output still showing | Streamlit cache | Press `C` in the app, or restart the server |
 
## ⚠️ Disclaimer
 
This is a model-based estimate, not a guarantee. Validate the model, class labels, calibration and decision threshold before using it for real credit or financial decisions. The app is intended for learning and demonstration purposes.
 
## 🔮 Future Improvements
 
- [ ] Add model performance metrics (accuracy, precision, recall, ROC-AUC)
- [ ] Show feature importance or SHAP explanations for each prediction
- [ ] Support batch predictions via CSV upload
- [ ] Add an adjustable decision threshold
- [ ] Deploy on Streamlit Community Cloud
## 🤝 Contributing
 
Contributions are welcome.
 
1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request
## 📄 License
 
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
 
## 👤 Author
 
**Mahesh**
 ---
 
⭐ If you found this project helpful, please give it a star!
 
