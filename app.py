
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.title("💳 Credit Scoring Dashboard")

# -----------------------------
# Step 1: Load saved models and WOE mappings
# -----------------------------
lr_model = joblib.load("logistic_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
woe_util = joblib.load("woe_util.pkl")
woe_delay = joblib.load("woe_delay.pkl")

# -----------------------------
# Step 2: User input (only critical features)
# -----------------------------
st.sidebar.header("Enter Customer Details")

LIMIT_BAL = st.sidebar.number_input("Credit Limit", min_value=1000, step=1000, value=50000)
BILL_AMT6 = st.sidebar.number_input("Last Month Bill Amount", min_value=0, step=100, value=20000)
PAY_0 = st.sidebar.number_input("Repayment Delay Last Month", min_value=-2, max_value=6, step=1, value=0)

# -----------------------------
# Step 3: Map WOE for new input
# -----------------------------
def map_woe_value(value, woe_dict):
    for k,v in woe_dict.items():
        if value in k:
            return v
    return 0

util_woe = map_woe_value(BILL_AMT6 / LIMIT_BAL, woe_util)
delay_woe = map_woe_value(PAY_0, woe_delay)

# -----------------------------
# Step 4: Predict Credit Score (Logistic Regression)
# -----------------------------
X_new = np.array([[util_woe, delay_woe]])
pred_prob = lr_model.predict_proba(X_new)[:,1][0]

# Convert probability to score
base_score = 600
pdo = 50
odds_base = 1/50
odds = pred_prob/(1-pred_prob)
score = base_score + pdo * np.log(odds/odds_base)/np.log(2)

st.subheader("Predicted Credit Score")
st.metric("Score", int(score))
st.write("Probability of Default:", round(pred_prob, 2))

# -----------------------------
# Step 5: Display Graphs (Training Distribution)

# -----------------------------

# st.subheader("Credit Score Distribution (from training data)")

data = pd.read_csv("UCI_Credit_Card.csv")
data["utilisation"] = data["BILL_AMT6"]/data["LIMIT_BAL"]
data["avg_delay"] = data[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].mean(axis=1)

def map_woe_df(df, feature, woe_dict):
    return df[feature].apply(lambda x: map_woe_value(x, woe_dict))

data['util_woe'] = map_woe_df(data, 'utilisation', woe_util)
data['delay_woe'] = map_woe_df(data, 'avg_delay', woe_delay)

X_train = data[['util_woe','delay_woe']]
y_train = data['default.payment.next.month']
y_pred_train = lr_model.predict_proba(X_train)[:,1]

# Score conversion
odds_train = y_pred_train/(1-y_pred_train)
score_train = base_score + pdo*np.log(odds_train/odds_base)/np.log(2)

# fig, ax = plt.subplots()
# ax.hist(score_train, bins=20, color='skyblue')
# ax.set_xlabel("Credit Score")
# ax.set_ylabel("Frequency")
# st.pyplot(fig)
# st.subheader("Credit Score Distribution (from training data)")

# fig, ax = plt.subplots(figsize=(8,5))
# ax.hist(score_train, bins=20, color='skyblue', edgecolor='black')
# ax.set_title("Distribution of Credit Scores in Training Data", fontsize=14)
# ax.set_xlabel("Credit Score", fontsize=12)
# ax.set_ylabel("Number of Customers", fontsize=12)
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # Highlight user score
# ax.axvline(score, color='red', linestyle='--', linewidth=2, label=f"Your Score: {int(score)}")
# ax.legend()
# st.pyplot(fig)

# Default probability histogram
# st.subheader("Default Probability Distribution (Logistic Regression)")
# fig2, ax2 = plt.subplots()
# ax2.hist(y_pred_train, bins=20, color='salmon')
# ax2.set_xlabel("Probability of Default")
# ax2.set_ylabel("Frequency")
# st.pyplot(fig2)
# st.subheader("Default Probability Distribution (Logistic Regression)")

# fig2, ax2 = plt.subplots(figsize=(8,5))
# ax2.hist(y_pred_train, bins=20, color='salmon', edgecolor='black')
# ax2.set_title("Distribution of Default Probabilities (Training Data)", fontsize=14)
# ax2.set_xlabel("Probability of Default", fontsize=12)
# ax2.set_ylabel("Number of Customers", fontsize=12)
# ax2.grid(axis='y', linestyle='--', alpha=0.7)

# # Highlight user probability
# ax2.axvline(pred_prob, color='green', linestyle='--', linewidth=2, label=f"Your Probability: {round(pred_prob,2)}")
# ax2.legend()
# st.pyplot(fig2)

# -----------------------------
# Step 6: XGBoost Feature Importance
# -----------------------------

# st.subheader("XGBoost Feature Importance")

features = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
            'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
            'SEX','EDUCATION','MARRIAGE']

# Use training data for feature importance plot
X_full = data[features]
y_full = data['default.payment.next.month']

y_pred_proba_xgb = xgb_model.predict_proba(X_full)[:,1]
auc_xgb = roc_auc_score(y_full, y_pred_proba_xgb)
# st.write(f"XGBoost AUC: {auc_xgb:.3f}")

# fig3, ax3 = plt.subplots(figsize=(8,6))
# xgb.plot_importance(xgb_model, max_num_features=10, ax=ax3)
# st.pyplot(fig3)
# st.subheader("Top 10 Important Features (XGBoost)")

# fig3, ax3 = plt.subplots(figsize=(8,6))
# xgb.plot_importance(xgb_model, max_num_features=10, ax=ax3, importance_type='weight', color='lightgreen')
# ax3.set_title("Top 10 Feature Importance (XGBoost)", fontsize=14)
# ax3.set_xlabel("F Score (Number of Times Feature Used in Splits)", fontsize=12)
# ax3.set_ylabel("Feature", fontsize=12)
# st.pyplot(fig3)

