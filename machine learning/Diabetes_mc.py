import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Import data
from sklearn.datasets import load_diabetes
df=load_diabetes()
print(df.DESCR)
target=df['target']
data=pd.DataFrame(data=df['data'],columns=df['feature_names'])
features=['age','sex','bmi','bp','tc','ldl','hdl','tch','ltg','glu']
data=pd.DataFrame(data=df['data'],columns=features)
data.head()
data.describe()

# Exploratory data
data['target']=target
corr=data.corr()
sns.heatmap(corr,annot=True)

ax=sns.boxplot(data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


q_25=data.quantile(0.25)
q_75=data.quantile(0.75)
IQR=q_75-q_25
upper=q_75+1.5*IQR
lower=q_25-1.5*IQR


data_c=data[~((data>upper)\
              |(data<lower)).any(axis=1)]
# Or
ind=((data>upper) | (data<lower)).any(axis=1)
data_cx=data[~ind]

ax=sns.boxplot(data_c.drop('target',axis=1))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

# Visualisation 
fig,ax=plt.subplots(figsize=(10,8))
plt.scatter(data_c['glu'],data_c['target'])
plt.xlabel('glu')
plt.ylabel('Disease progression after 1 year baseline')

## Build regression models
from sklearn.model_selection import train_test_split

X=data_c.drop('target',axis=1)
y=data_c['target']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# Simple linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

lm=LinearRegression().fit(x_train,y_train) 
print('Training score:',lm.score(x_train,y_train)) ## R-square
predictors=x_train.columns

coef=pd.Series(lm.coef_,predictors).sort_values()
print(coef)
y_pred=lm.predict(x_test)
pd_actual=pd.DataFrame({'actual':y_test,'predicted':y_pred,'sex':x_test['sex']})
print('Testing score:',r2_score(y_test,y_pred))
# scater plot
fig, ax=plt.subplots()
sns.scatterplot(data=pd_actual,x='actual',y='predicted')
ax.plot(np.linspace(min(y_test), max(y_test), 100), np.linspace(min(y_test), max(y_test), 100), color='black', linestyle='--')
ax.set_xlabel('diabete progression measured')
ax.set_ylabel('diabete progression predicted')
r2 = r2_score(y_test, y_pred)
ax.text(min(pd_actual['actual']),max(pd_actual['predicted']), f'R2 = {r2:.2f}',fontsize=12)

sns.lmplot(data=pd_actual,x='actual',y='predicted',hue='sex')
