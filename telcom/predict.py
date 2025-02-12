import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
print(df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,9))
sns.countplot(x="Contract",hue="Churn",data=df,palette="coolwarm")
plt.title("Churn Distribution by Contract Type")
plt.xticks(rotation=45)
plt.show()


