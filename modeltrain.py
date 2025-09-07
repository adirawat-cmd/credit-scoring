from sqlalchemy import create_engine,text
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
engine = create_engine("your sql credentials")

# with engine.connect() as conn:
#     print(conn.execute(text("SELECT version();")).fetchone())

df = pd.read_csv("data/UCI_Credit_Card.csv")

df.to_sql("credit_data",engine,if_exists="replace",index=False)

# data = pd.read_sql("SELECT * FROM credit_data LIMIT 5;",engine)
# print(data)

data = pd.read_sql("SELECT * FROM credit_data",engine)
data["utilisation"] = data["BILL_AMT6"]/data["LIMIT_BAL"]
data["avg_delay"] = data[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].mean(axis=1)
data['max_delay'] = data[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].max(axis=1)

data["util_bin"] = pd.qcut(data["utilisation"],5)
default_rate_table = data.groupby("util_bin")["default.payment.next.month"].mean()
print(default_rate_table)
data['delay_bin'] = pd.qcut(data['avg_delay'], 5)
default_rate_delay = data.groupby('delay_bin')['default.payment.next.month'].mean()
print(default_rate_delay)
woe_table = data.groupby('util_bin')['default.payment.next.month'].agg(['count','sum'])
woe_table['non_event'] = woe_table['count'] - woe_table['sum']
woe_table['event_rate'] = woe_table['sum'] / woe_table['sum'].sum()
woe_table['non_event_rate'] = woe_table['non_event'] / woe_table['non_event'].sum()
woe_table['woe'] = np.log(woe_table['non_event_rate'] / woe_table['event_rate'])
woe_table['IV'] = (woe_table['non_event_rate'] - woe_table['event_rate']) * woe_table['woe']
woe_table = woe_table.reset_index()
print(woe_table[['util_bin','woe','IV']])
print("Total IV:", woe_table['IV'].sum())
woe_table_delay = data.groupby('delay_bin')['default.payment.next.month'].agg(['count','sum'])
woe_table_delay['non_event'] = woe_table_delay['count'] - woe_table_delay['sum']
woe_table_delay['event_rate'] = woe_table_delay['sum'] / woe_table_delay['sum'].sum()
woe_table_delay['non_event_rate'] = woe_table_delay['non_event'] / woe_table_delay['non_event'].sum()
woe_table_delay['woe'] = np.log(woe_table_delay['non_event_rate'] / woe_table_delay['event_rate'])
woe_table_delay['IV'] = (woe_table_delay['non_event_rate'] - woe_table_delay['event_rate']) * woe_table_delay['woe']
woe_table_delay = woe_table_delay.reset_index()
print(woe_table_delay[['delay_bin','woe','IV']])
print("Total IV:", woe_table_delay['IV'].sum())

# Map WOE values back to the dataframe
woe_map_util = dict(zip(woe_table['util_bin'], woe_table['woe']))
data['util_woe'] = data['util_bin'].map(woe_map_util)

woe_map_delay = dict(zip(woe_table_delay['delay_bin'], woe_table_delay['woe']))
data['delay_woe'] = data['delay_bin'].map(woe_map_delay)

x = data[["util_woe","delay_woe"]]
y = data["default.payment.next.month"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(roc_auc_score(y_test,y_pred))

print(y_pred)
for col, coef in zip(x.columns, model.coef_[0]):
    print(col, coef)



features = ['LIMIT_BAL','AGE',
            'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
            'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
            'SEX','EDUCATION','MARRIAGE']

X = data[features]
y = data['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_proba = xgb_clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_proba)
print("XGBoost AUC:", auc)

xgb.plot_importance(xgb_clf,max_num_features = 10)
plt.show()


p = model.predict_proba(x_test)  # probability of default
odds = p / (1 - p)
base_score = 600
pdo = 50
odds_base = 1/50

score = base_score + pdo * np.log(odds / odds_base) / np.log(2)

plt.hist(score, bins=20)
plt.title("Credit Score Distribution")
plt.show()

# Save models
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(xgb_clf, "models/xgb_model.pkl")

# Save WOE mappings too
joblib.dump(woe_map_util, "models/woe_util.pkl")
joblib.dump(woe_map_delay, "models/woe_delay.pkl")
