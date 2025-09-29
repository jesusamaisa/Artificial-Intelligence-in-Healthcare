# import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

# load dataset

df = pd.read_csv("C://Users//muham//OneDrive//Desktop//ZeroToHero//Sağlıkta Yapay Zeka Serisi//Kronik Böbrek Hastalığı Tahmini//kidney_disease.csv")
df.drop("id", axis = 1, inplace = True)

df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 
              'red_blood_cells', 'pus_cell', 'pus_cell_clumbs', 
              'bacteria', 'blood_glucose_random', 'blood_urea',
           'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 
           'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count', 
           'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
           'appetite', 'peda_edema', 'aanemia', 'class']

df.info()

describe = df.describe()

df["packed_cell_volume"] = pd.to_numeric(df["packed_cell_volume"], errors="coerce")
df["white_blood_cell_count"] = pd.to_numeric(df["white_blood_cell_count"], errors="coerce")
df["red_blood_cell_count"] = pd.to_numeric(df["red_blood_cell_count"], errors="coerce")

# EDA: KDE

cat_cols = [col for col in df.columns if df[col].dtype == "object"] # categoric veri
num_cols = [col for col in df.columns if df[col].dtype != "object"] # float, int -> numeric

for col in cat_cols:
    print(f"{col}: {df[col].unique()}")
"""
diabetes_mellitus: ['yes' 'no' ' yes' '\tno' '\tyes' nan]
coronary_artery_disease: ['no' 'yes' '\tno' nan]
class: ['ckd' 'ckd\t' 'notckd']
"""
df["diabetes_mellitus"].replace(to_replace = {'\tno':"no", '\tyes': "yes", ' yes':"yes"}, inplace=True)
df["coronary_artery_disease"].replace(to_replace = {'\tno':"no"}, inplace=True)
df["class"].replace(to_replace = {'ckd\t':"ckd"}, inplace=True)

df["class"] = df["class"].map({"ckd":0, "notckd":1})

plt.figure(figsize = (15,15))
plotnumber = 1

for col in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3 , 5, plotnumber)
        sns.distplot(df[col])
        plt.xlabel(col)
        
    plotnumber += 1

plt.tight_layout()
plt.show()

plt.figure()
sns.heatmap(df.corr(), annot = True, linecolor="white", linewidths=2)
plt.show()

def kde(col):
    grid = sns.FacetGrid(df, hue="class", height=6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
 
kde("hemoglobin")
kde("white_blood_cell_count")
kde("packed_cell_volume")
kde("red_blood_cell_count")
kde("albumin")
kde("specific_gravity")

# Preprocessing: missing value problem

df.isna().sum().sort_values(ascending=False)

def solve_mv_random_value(feature):
    
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

for col in num_cols:
    solve_mv_random_value(col)

df[num_cols].isnull().sum()

def solve_mv_mode(feature):
    
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
    
solve_mv_random_value("red_blood_cells")
solve_mv_random_value("pus_cell") 
    
for col in cat_cols:
    solve_mv_mode(col)
    
df[cat_cols].isnull().sum()    

# Preprocessing: Feature encoding

for col in cat_cols:
    print(f"{col}: {df[col].nunique()} ")

encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Model (DT) training and testing

independent_col = [col for col in df.columns if col != "class"] # X
dependent_col = "class" # y

X = df[independent_col]
y = df[dependent_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print("Confusion matrix: \n",cm)
print("classification_report: \n",cr)

# DT visualization - feature importance

class_names = ["ckd", "notckd"]

plt.figure(figsize=(20,10))
plot_tree(dtc, feature_names=independent_col, filled=True, rounded=True, fontsize=8)
plt.show()

feature_importance = pd.DataFrame({"Feature":independent_col, "Importance":dtc.feature_importances_})

print("Most important feature: ",feature_importance.sort_values(by="Importance",ascending=False).iloc[0])

plt.figure()
sns.barplot(x = "Importance", y = "Feature", data = feature_importance)
plt.title("Feature Importance")
plt.show()
