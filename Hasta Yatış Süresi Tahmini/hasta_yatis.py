#kütüphaneleri import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

# Veri setini yükle
df = pd.read_csv("C://Users//muham//OneDrive//Desktop//ZeroToHero//Sağlıkta Yapay Zeka Serisi//Hasta Yatış Süresi//veriseti.csv")
df_ = df.head(50)

df.info()

describe = df.describe()

los = df['Length of Stay'].value_counts()
df['Length of Stay']= df ['Length of Stay'].replace("120+",120)
df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')
los = df['Length of Stay'].value_counts()

df.isna().sum()

for column in df.columns:
    unique_values = df[column].nunique()
    print(f"{column}: {unique_values} unique values")

df = df[df["Patient Disposition"] != 'Expired']

# EDA (Keşifsel Veri Analizi)

"""
    hasta_yatis_suresi - age - type of admission - payment type 
"""
sns.boxplot(x='Payment Typology 1', y='Length of Stay', data=df)
plt.title('Payment Typology 1 vs Length of Stay')
plt.xticks(rotation=60)
plt.show()

sns.countplot(x= "Age Group", data=df[df["Payment Typology 1"] == 'Medicare'], order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"])
plt.title('Medicare Partients for Age Group')
plt.show()

sns.boxplot(x='Type of Admission', y='Length of Stay', data=df)
plt.title('Type of Admission 1 vs Length of Stay')
plt.xticks(rotation=60)
plt.show()

f, ax = plt.subplots()
sns.boxplot(x='Age Group', y='Length of Stay', data=df , order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"])
plt.title('Age Group vs Length of Stay')
plt.xticks(rotation=60)
ax.set(ylim = (0, 25))
plt.show()

# Feature Encoding (Özellik Kodlama)
df = df.drop(["Hospital Service Area","Hospital County","Operating Certificate Number",
              "Facility Name","Zip Code - 3 digits","Patient Disposition","CCSR Diagnosis Description",
              "Discharge Year","CCSR Procedure Description","APR DRG Description","APR MDC Description",
              "Payment Typology 2","Payment Typology 3","Birth Weight","Total Charges","Total Costs"
              ], axis=1)

age_group_index = {"0 to 17": 1, "18 to 29": 2, "30 to 49": 3, "50 to 69": 4, "70 or Older": 5}
gender_index = {"U":0,"F": 1, "M": 2}
risk_and_severity_index = {np.nan:0 ,"Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}

df["Age_Group"] = df["Age Group"].apply(lambda x: age_group_index[x])
df["Gender"] = df["Gender"].apply(lambda x: gender_index[x])
df["APR Risk of Mortality"] = df["APR Risk of Mortality"].apply(lambda x: risk_and_severity_index[x])


encoder = OrdinalEncoder()
df["Race"] = encoder.fit_transform(np.asarray(df["Race"]).reshape(-1, 1))
df["Ethnicity"] = encoder.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1, 1))
df["Type of Admission"] = encoder.fit_transform(np.asarray(df["Type of Admission"]).reshape(-1, 1))
df["CCSR Diagnosis Code"] = encoder.fit_transform(np.asarray(df["CCSR Diagnosis Code"]).reshape(-1, 1))
df["CCSR Procedure Code"] = encoder.fit_transform(np.asarray(df["CCSR Procedure Code"]).reshape(-1, 1))
df["APR Medical Surgical Description"] = encoder.fit_transform(np.asarray(df["APR Medical Surgical Description"]).reshape(-1, 1))
df["Payment Typology 1"] = encoder.fit_transform(np.asarray(df["Payment Typology 1"]).reshape(-1, 1))
df["Emergency Department Indicator"] = encoder.fit_transform(np.asarray(df["Emergency Department Indicator"]).reshape(-1, 1))

# Missing Values (Eksik Değerler)

df.isna().sum()

df = df.drop("CCSR Procedure Code", axis=1)
df = df.dropna(subset=["Permanent Facility Id","CCSR Diagnosis Code"])

# Train Test Split (Eğitim Test Bölme)

x = df.drop("Length of Stay", axis=1)
y = df["Length of Stay"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Regression: train ve test

dtree = DecisionTreeRegressor()
dtree.fit(x_train, y_train)
train_prediction = dtree.predict(x_train)
test_prediction = dtree.predict(x_test)

print("RMSE: Train" , np.sqrt(mean_squared_error(y_train,train_prediction)))
print("RMSE: Test" , np.sqrt(mean_squared_error(x_test,test_prediction)))

"""
overfitting
RMSE: Train:  2.84783327422551 -> 7 - 10 - 13
RMSE: Test:  7.976502723219912 -> 2 - 10 - 18

after max_depth = 10
RMSE: Train:  6.088278470926022
RMSE: Test:  6.242028697402208
"""


# Kategorik hale getirme: Classification


bins = [0, 5, 10, 20, 30, 50, 120]
labels = [5, 10, 20, 30, 50, 120]

df["los_bin"] = pd.cut(x=df["Length of Stay"], bins=bins)
df["los_label"] = pd.cut(x=df["Length of Stay"], bins=bins, labels=labels)
df_ = df.head(50)
df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace(","," -"))
df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace("120","120+"))

f, ax = plt.subplots()
sns.countplot(x="los_bin", data=df)

new_X = df.drop(["Length of Stay", "los_bin", "los_label"], axis = 1)
new_y = df["los_label"]

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size = 0.2, random_state = 42)

dtree = DecisionTreeClassifier(max_depth=10)
dtree.fit(X_train, y_train)

train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print("Train Accuracy: ", accuracy_score(y_train, train_prediction))
print("Test Accuracy: ", accuracy_score(y_test, test_prediction))
print("Classification report: ", classification_report(y_test, test_prediction))

"""
Overfitting
Train Accuracy:  0.9244704097809807
Test Accuracy:  0.6851298279739233

after max_depth = 10
Train Accuracy:  0.7418194070663043
Test Accuracy:  0.7410697755934764
"""





