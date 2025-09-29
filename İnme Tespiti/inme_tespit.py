# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
"""
standartScaler
DecisionTree
Pipeline(standartScaler, DecisionTree)
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# load and EDA
df = pd.read_csv("İnme Tespiti//healthcare-dataset-stroke-data.csv")
df = df.drop(["id"], axis = 1)

df.info()

describe = df.describe()

# stroke etiketinin dagilimi
plt.figure()
sns.countplot(x="stroke", data=df)
plt.title("Distribution of Stroke Class")
plt.show()

"""
Dengesiz veri seti
4800 -> 0
250 -> 1

kcy: tum sonuclara 0 de Acc: 4800/(5100) = 0.94 Yaniltici sonuc.
Yanilmamak icin:
    - cm
    - f1 score
Dengesiz veri seti cozumu:
    - stroke (1) sayisini arttırmamiz lazim, veri toplama
    - down sampling (0) sayisini azalt, veri kaybi olur         
"""

# Missing Value: DecisionTreeRegressor
df.isnull().sum()

DT_bmi_pipe = Pipeline(steps=[
    ("scale", StandardScaler()), # veriyi standartlastirmak icin standart scaler
    ("dtr", DecisionTreeRegressor()) # karar agaci regresyon modeli
    ]) 

X = df[["gender", "age", "bmi"]].copy()

# gender sutununda bulunan degerleri sayisal degerlere donustuyoruz
# male -> 0, female -> 1, other -> -1
X.gender = X.gender.replace({"Male": 0, "Female":1, "Other":-1}).astype(np.uint8)

# bmi degeri eksik olan nan satirlari ayir
missing = X[X.bmi.isna()]

# bmi degeri eksik olmayan verileri ayiralim
X = X[~X.bmi.isna()]
y = X.pop("bmi")

# modeli eksik olmayan veriler ile egit
DT_bmi_pipe.fit(X,y)

# eksik bmi degelerini tahmin edelim, tahmin yapilirken gender ve age kullabilacak
predicted_bmi = pd.Series(DT_bmi_pipe.predict(missing[["gender", "age"]]), index=missing.index)

df.loc[missing.index, "bmi"] = predicted_bmi

# Model prediction: encoding, training and testing
df["gender"] = df["gender"].replace({"Male": 0, "Female":1, "Other":-1}).astype(np.uint8)
df["Residence_type"] = df["Residence_type"].replace({"Rural": 0, "Urban":1}).astype(np.uint8)
df["work_type"] = df["work_type"].replace({"Private": 0, "Self-employed":1, "Govt_job":2, "children":-1, "Never_worked":-2}).astype(np.uint8)

X = df[["gender","age", "hypertension","heart_disease","work_type","avg_glucose_level", "bmi"]]
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

logreg_pipe = Pipeline(steps=[("scale", StandardScaler()), ("LR", LogisticRegression())])

# model training
logreg_pipe.fit(X_train, y_train)

# modelin testi
y_pred = logreg_pipe.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Cm: \n", confusion_matrix(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))
"""
50
 [[20   5]
 [ 5   20]]
"""
# model kaydetme ve geri yukleme, gerçek hasta testi (olasilik goster - %90 -> 1, %10 -> 0)

import joblib

# # modeli kaydetme
# joblib.dump(logreg_pipe, "log_reg_model.pkl")

# %% model yukleme

loaded_log_reg_pipe = joblib.load("İnme Tespiti//log_reg_model.pkl")

# yeni hasta verisi tahmin etme
new_patient_data = pd.DataFrame({
    "gender":[1],
    "age":[45],
    "hypertension":[1],
    "heart_disease":[0],
    "work_type":[0],
    "avg_glucose_level":[70],
    "bmi" : [25]
    })

# tahmin
new_patient_data_result = loaded_log_reg_pipe.predict(new_patient_data)

# tahmin olasiliksal
new_patient_data_result_probability = loaded_log_reg_pipe.predict_proba(new_patient_data)















