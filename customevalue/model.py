import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import shap
from flask import Flask,request,jsonify
import pickle

param_grid_rf = {
    "n_estimators": np.arange(50, 201, 50),
    "max_depth": np.arange(3, 15, 1),
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}
param_grid_xgb = {
    "n_estimators": np.arange(50, 201, 50),
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3]
}
df=pd.read_csv("train.csv")

df["Sex"]=df["Sex"].map({"male":0,"female":1})
df["Age"].fillna(df["Age"].median(),inplace=True)
features=["class", "Sex", "Age", "status"]
x = df[features]
y=df["customvalue"]

x_train,x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

xgb_model=XGBClassifier(enable_categorical=True,eval_metric="logloss")
xgb_model.fit(x_train,y_train)

y_pred_rf = rf_model.predict(x_test)
y_pred_log = log_reg.predict(x_test)
y_pred_xgb = xgb_model.predict(x_test)

random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid_rf, 
                                      n_iter=10, cv=3, random_state=42, n_jobs=-1)
random_search_rf.fit(x_train, y_train)
print("Best Hyperparameters for Random Forest:", random_search_rf.best_params_)

# Evaluate on the test set
best_rf_model = random_search_rf.best_estimator_
y_pred_rf_tuned = best_rf_model.predict(x_test)
print(f"Tuned Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf_tuned):.4f}")

random_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid_xgb,
                                       n_iter=10, cv=3, random_state=42)
random_search_xgb.fit(x_train, y_train)

# Best hyperparameters
print("Best Hyperparameters for XGBoost:", random_search_xgb.best_params_)

# Evaluate on the test set
best_xgb_model = random_search_xgb.best_estimator_
y_pred_xgb_tuned = best_xgb_model.predict(x_test)
print(f"Tuned XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb_tuned):.4f}")

rf_acc = accuracy_score(y_test, y_pred_rf)
log_reg_acc = accuracy_score(y_test, y_pred_log)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print(classification_report(y_test, y_pred_xgb))
print(f"Logistic Regression Accuracy: {log_reg_acc:.4f}")
print(classification_report(y_test, y_pred_log))
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, y_pred_rf))
model_results = {
    "Logistic Regression": log_reg_acc,
    "Random Forest": rf_acc,
    "XGBoost": xgb_acc
}

# Print model accuracies
for model, acc in model_results.items():
    print(f"{model}: {acc:.4f}")
importances_rf=best_rf_model.feature_importances_
feature_importance_df_rf = pd.DataFrame({
    "Feature": x.columns,
    "Importance": importances_rf
}).sort_values(by="Importance", ascending=False)

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Create a figure with two subplots for SHAP summary plots
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Create SHAP explainer for Random Forest
explainer_rf = shap.TreeExplainer(best_rf_model)
shap_values_rf = explainer_rf.shap_values(x_test)

plt.subplot(1, 2, 1)
shap.summary_plot(shap_values_rf, x_test, show=False)
plt.title("SHAP Summary - Random Forest")

# Create SHAP explainer for XGBoost
explainer_xgb = shap.TreeExplainer(best_xgb_model)
shap_values_xgb = explainer_xgb.shap_values(x_test)

plt.subplot(1, 2, 2)
shap.summary_plot(shap_values_xgb, x_test, show=False)
plt.title("SHAP Summary - XGBoost")


# Create a figure with two subplots for feature importance


# Plot feature importance for Random Forest
sns.barplot(x="Importance", y="Feature", data=feature_importance_df_rf, palette="viridis", ax=axes[0])
axes[0].set_title("Feature Importance - Random Forest")

# Plot feature importance for XGBoost
xgb.plot_importance(best_xgb_model, importance_type='weight', max_num_features=10, height=0.5, ax=axes[1])
axes[1].set_title("Feature Importance - XGBoost")

plt.tight_layout()
plt.show()

pickle.dump(best_rf_model, open("tuned_rf_model.pkl", "wb"))
pickle.dump(best_xgb_model, open("tuned_xgb_model.pkl", "wb"))
# log_reg=LogisticRegression(max_iter=1000)
# log_reg.fit(x_train,y_train)
# y_pred_log=log_reg.predict(x_test)
# log_reg_acc=accuracy_score(y_test,y_pred_log)
# print(f"Logistic Regression Accuracy:{log_reg_acc:.4f}")
# print(classification_report(y_test, y_pred_log))
# print(f"Training samples: {x_train.shape[0]}")
# print(f"Testing samples: {x_test.shape[0]}")
# df=pd.read_csv("train.csv")
# print(f"missing value\n{df.isnull().sum()}")
# df["Age"].fillna(df["Age"].median(),inplace=True)
# df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
# df.drop(columns=["Cabin"], inplace=True)
# print(f"missing value\n{df.isnull().sum()}")
# plt.figure(figsize=(10,6))

# sns.countplot(x="Sex",hue="Survived",data=df,palette="Set2")
# plt.title("Survival by Gender")
# plt.show()
# survival_rate=df["Survived"].mean()*100
# print(f"Survival Rate:{survival_rate:.2f}%")

# sns.histplot(df[df["Survived"]==1]["Age"],kde=True,color="green", label="Survived",bins=30)
# sns.histplot(df[df["Survived"]==0]["Age"],kde=True,color="red", label="Not Survived",bins=30)
# plt.legend()
# plt.title("Age Distribution by Survival Status")
# plt.xlabel("Age")
# plt.ylabel("Count")

# sns.countplot(x="Pclass",hue="Survived",data=df,palette="Set1")
# plt.show()

# df["FamilySize"]=df["SibSp"]+df["Parch"]+1
# sns.countplot(x="FamilySize", hue="Survived", data=df, palette="muted")
# plt.title("Survival Rate by Family Size")
# plt.show()

# Correlation heatmap
# corr = df.select_dtypes(include=['number']).corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap")
# plt.show()
