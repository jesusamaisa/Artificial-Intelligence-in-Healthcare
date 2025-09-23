#   Kütüphaneleri yükliyoruz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

import warnings
warnings.filterwarnings("ignore")
#   Veri setimizi yüklüyoruz ve ön inceleme yapıyoruz
df = pd.read_csv("Kalp Hastalığı Tahmini//heart_disease_uci.csv")
df = df.drop(columns = ["id"])

df.info()

describe = df.describe()

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure()
sns.pairplot(df,vars=numerical_features ,hue="num")
plt.show()

plt.figure()
sns.countplot(x='num', data=df)
plt.show()

#   Missing value kontrolü yapıyoruz

print(df.isnull().sum())
df = df.drop(columns=["ca"])
print(df.isnull().sum())

df["trestbps"].fillna(df["trestbps"].median(), inplace=True)
df["chol"].fillna(df["chol"].median(), inplace=True)
df["fbs"].fillna(df["fbs"].mode()[0], inplace=True)
df["restecg"].fillna(df["restecg"].mode()[0], inplace=True)
df["thalch"].fillna(df["thalch"].median(), inplace=True)
df["exang"].fillna(df["exang"].mode()[0], inplace=True)
df["oldpeak"].fillna(df["oldpeak"].median(), inplace=True)
df["slope"].fillna(df["slope"].mode()[0], inplace=True)
df["thal"].fillna(df["thal"].mode()[0], inplace=True)

print(df.isnull().sum())

#   Traşn test split işlemi yapıyoruz

X = df.drop(columns=["num"], axis=1)
y = df["num"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

categorial_features = ["sex", "dataset", "cp", "restecg", "exang", "slope", "thal"]
numerical_features = ["age", "trestbps", "chol", "fbs", "thalch", "oldpeak"]

x_train_num = x_train[numerical_features]
x_test_num = x_test[numerical_features]

scaler = StandardScaler()
x_train_num_scaled = scaler.fit_transform(x_train_num)
x_test_num_scaled =scaler.transform(x_test_num)

encoder = OneHotEncoder(sparse_output=False,drop="first")
x_train_cat = x_train[categorial_features]
x_test_cat = x_test[categorial_features]

x_train_cat_encoded = encoder.fit_transform(x_train_cat)
x_test_cat_encoded = encoder.transform(x_test_cat)

x_train_traformed = np.hstack((x_train_num_scaled, x_train_cat_encoded))
x_test_traformed = np.hstack((x_test_num_scaled, x_test_cat_encoded))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier()

voting_clf = VotingClassifier(estimators=[('rf', rf), ('knn', knn)], voting='soft')

# Modeli eğitme
voting_clf.fit(x_train_traformed, y_train)

# Tahmin yapma
y_pred = voting_clf.predict(x_test_traformed)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# CM
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


#   Standartlaştırma yapıyoruz

#   Kategorik kodlama (label encoding) yapıyoruz

#   Modelleme yapıyoruz : RF, KNN, Voting Classifier train ve test

#   CM (confusion matrix)

