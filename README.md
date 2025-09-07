# credit-scoring
Link: https://credit-scoring-6omlurkcyomuocduqey6tl.streamlit.app/
💳 Credit Scoring Dashboard

A machine learning–powered credit scoring system built with Python (Scikit-learn, XGBoost, SQL skills) and deployed using Streamlit.
This app predicts a credit score and probability of default for customers based on financial and behavioral data.

📌 Features

Logistic Regression & XGBoost models trained on the UCI Credit Card Dataset

Probability of Default (PD) calculation with score transformation

Clean Streamlit interface for customer inputs

Displays credit score, default probability, and risk category

(Optional) Graphs for:

Credit score distribution

Default probability histogram

XGBoost feature importance

🛠️ Tech Stack

Python 3.9+

Scikit-learn – Logistic Regression, feature scaling, model evaluation

XGBoost – Gradient boosting classifier for feature importance and AUC scoring

Joblib – Model persistence

Pandas / NumPy – Data handling

Matplotlib – Visualizations (for explainability version)

Streamlit – Interactive dashboard deployment

📂 Project Structure
credit-scoring-app/
│── data/
│   └── UCI_Credit_Card.csv
│── models/
│   ├── logistic_model.pkl
│   ├── xgb_model.pkl
│   ├── woe_util.pkl
│   └── woe_delay.pkl
│── app.py               # Minimal deployment app (no graphs)
│── app_explained.py     # Full version with graphs and analysis
│── requirements.txt
│── README.md

⚙️ Installation

Clone the repo:

git clone https://github.com/adirawat-cmd/credit-scoring
cd credit-scoring-app

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

Install dependencies:
pip install -r requirements.txt

🚀 Run the App
streamlit run app.py
📊 Example Output
Credit Score: 720
Probability of Default: 0.18 (18%)
Risk Category: Low Risk

🧪 Model Performance
Logistic Regression AUC: ~0.65
XGBoost AUC: ~0.833

📖 Dataset
The app is trained on the UCI Credit Card Dataset:
Default of Credit Card Clients Dataset (UCI Machine Learning Repository)

🔮 Future Improvements
Add SHAP explainability for model transparency
Integrate with a SQL backend for customer data storage
Deploy on Streamlit Cloud / AWS / Heroku


👉 This README makes your repo look professional, demo-ready, and deployable.

Do you want me to also prepare a requirements.txt so anyone can directly run your Streamlit app after cloning?
