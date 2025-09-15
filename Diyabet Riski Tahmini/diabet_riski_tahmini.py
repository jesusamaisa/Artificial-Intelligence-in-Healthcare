#
# Kütüphaneleri yükleyeceğiz
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")
#
# Veri yüklücez ve EDA (keşifsel veri analizi)
#
df = pd.read_csv("Diyabet Riski Tahmini//diabetes.csv")
df_name = df.columns

df.info()
#sutün isimleri (büyük küçük harf boşluk ingilizce olmayan karakterşer)
# sample sayısı ve kayıp veri problemi
#veri tiplerinin düzgünlüğü Örnek: Age ---> String olamaz gibi

df.describe()

# sns.pairplot(df,hue="Outcome")
# plt.show()

def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True, cmap= "coolwarm", fmt=".2f",linewidths=0.5)
    plt.title("Correlation of Features")
    plt.show()

plot_correlation_heatmap(df)
#
# Ouitler (aykırı) tespiti
#

#IQR NEDİR? 
#IQR, işletmelerde gelir oranlarını gösteren bir gösterge olarak kullanılır. 
#Simetrik bir dağılım için ( medyan , ortanca değere, yani birinci ve üçüncü çeyreklerin ortalamasına 
#eşit olduğunda), IQR'nin yarısı medyan mutlak sapmasına (MAD) eşittir. 
#Ortanca , merkezi eğilimin karşılık gelen ölçüsüdür.
def detect_outliers_iqr(df):
    outlier_indices = []
    outliers_df = pd.DataFrame()

    for col in df.select_dtypes(include=["float64","int64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1 #interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_in_col = df[(df[col]<lower_bound) | (df[col] >upper_bound)]
        outlier_indices.extend(outlier_in_col.index)
        outliers_df = pd.concat([outliers_df,outlier_in_col],axis=0)

# remove duplicate indices
    outlier_indices = list(set(outlier_indices))

#remove duplicate rows in the ouitlers dataframe
    outliers_df = outliers_df.drop_duplicates()

    return outliers_df,outlier_indices

outliers_df,outlier_indices = detect_outliers_iqr(df)

#remove outliers from the original dataframe
df_cleaned = df.drop(outlier_indices).reset_index(drop=True)


#
# Eğitim ve Test ayrımı
#
x = df_cleaned.drop(["Outcome"],axis=1)
y = df_cleaned["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)


#
# Standartlaştırma
#

# Veri setimizin içinde farklı ölçeklerde değişkenler olabilir.  Bazı değerlerimiz
# 0-1 arasında iken bazıları 0-1000 arasında olabilir. Bu durumda modelimiz büyük
# ölçekli değişkenlere daha fazla önem verecektir. Bu yüzden tüm değişkenleri aynı
# ölçeğe getirmemiz gerekir. Bunun için standartlaştırma (standardization) yaparız.

scaler = StandardScaler()

scaler.fit_transform(x_train)

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)




#
# Model eğitimi ve değerlendirilmesi
#
"""
LogisticRegression
DecisionTreeClassifier
KNeighborsClassifier
GaussianNB
SVC
AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
"""
def getBasedModel():
    basedModels = []
    basedModels.append(("LR",LogisticRegression()))
    basedModels.append(("DT",DecisionTreeClassifier()))
    basedModels.append(("KNN",KNeighborsClassifier()))
    basedModels.append(("NB",GaussianNB()))
    basedModels.append(("SVM",SVC()))
    basedModels.append(("RF",RandomForestClassifier()))
    basedModels.append(("GBM",GradientBoostingClassifier()))
    basedModels.append(("AdaB",AdaBoostClassifier()))

    return basedModels

def baseModelsTraining(models,x_train,y_train):

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model,x_train,y_train,cv=kfold,scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy: {cv_results.mean()}, std: ({cv_results.std()})")
        
    return results,names

def plot_box(names, results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
models = getBasedModel()
results, names = baseModelsTraining(models, x_train_scaled, y_train)
plot_box(names, results)

#
# hyperparameter tuning
#

# DecionTreeClassifier için hyperparameter'lar

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier()

# GridSearchCV kullanarak en iyi hyperparameter'ları bulma
grid_search =GridSearchCV(estimator=dt, param_grid=param_grid,scoring='accuracy', cv=5, )

#traning
grid_search.fit(x_train_scaled, y_train)
print("En iyi parametreleri : ", grid_search.best_params_)

best_dt_model = grid_search.best_estimator_

y_pred = best_dt_model.predict(x_test)

print("Confusion Matrix:\n",)
print(confusion_matrix(y_test, y_pred))

"""
[[  0 109]
 [  0  51]]
"""
print("Classification_report:\n",)
print(classification_report(y_test, y_pred))

"""
Classification_report:

              precision    recall  f1-score   support

           0       0.00      0.00      0.00       109
           1       0.32      1.00      0.48        51

    accuracy                           0.32       160
   macro avg       0.16      0.50      0.24       160
weighted avg       0.10      0.32      0.15       160
"""

#
# Modelin gerçek veriyle test edilmesi
#

new_data = np.array([[6,148,72,35,0,34.6,0.627,51]])

new_prediction = best_dt_model.predict(new_data)

print("Yeni veri tahmini (0: Diyabet değil, 1: Diyabet): ", new_prediction)