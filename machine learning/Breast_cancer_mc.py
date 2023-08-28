import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
print(cancer.DESCR)
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
target=cancer['target']
cancer.target_names
target_names=cancer['target_names']
df.head()
df.describe()

## Generate data 
df.isnull().sum()
data=df.drop(['radius error','texture error','perimeter error','smoothness error',
             'compactness error','concavity error','concave points error','symmetry error',
             'fractal dimension error','area error'],axis=1)

data1=data.drop(['mean radius','worst radius','mean compactness','worst compactness',
                 'mean concavity','worst concavity','mean concave points','worst concave points'],axis=1)
# Calculate z-score 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_s=scaler.fit_transform(data)
data_s=pd.DataFrame(data=data_s, columns=data.columns)
data_s.head()

bp=sns.boxplot(data_s)
bp.set_xticklabels(bp.get_xticklabels(),rotation=90)

q_25=data_s.quantile(0.25)
q_75=data_s.quantile(0.75)
IQR=q_75-q_25
upper=q_75+1.5*IQR
lower=q_25-1.5*IQR
# Remove outliers
data_s['target']=target
data_c=data_s[~((data_s>upper)\
                |(data_s<lower)).any(axis=1)]
bp1=sns.boxplot(data_c.drop('target',axis=1))
bp1.set_xticklabels(bp1.get_xticklabels(),rotation=90)

## Machine learning_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X=data_c.drop('target',axis=1)
y=LabelEncoder.fit_transform(data_c,data_c['target'])
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

## Logistic regression classifier 
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(solver='liblinear').fit(x_train,y_train) 
y_pred=log.predict(x_test)
# Calculate accuracy on the test set
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {overall_accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_accuracy=[]
F1=[]
Recall=[]
Precision=[]
for i in np.unique(y_test):
    label=np.where(y_test == i)
    class_acc=accuracy_score(y_test[label],y_pred[label])
    class_accuracy.append(class_acc)
    fscore=f1_score(y_test[label],y_pred[label],average='micro')
    F1.append(fscore)
    re=recall_score(y_test[label],y_pred[label],average='micro')
    Recall.append(re)
    pre=precision_score(y_test[label],y_pred[label],average='micro')
    Precision.append(pre)


Results=pd.DataFrame({'Class accuracy':[class_accuracy],'F1':[F1],'Precision':[Precision],'Recall':[Recall]}).transpose()
print('Magilant vs Benign:', Results)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_fn=DecisionTreeClassifier(max_depth=None)
model=decision_fn.fit(x_train,y_train)
y_pred=model.predict(x_test)
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {overall_accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Randome forest 
from sklearn.ensemble import RandomForestClassifier
Random_fn=RandomForestClassifier(n_estimators=20,max_depth=None)
model=Random_fn.fit(x_train,y_train)
model.predict(x_test)

overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {overall_accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)







