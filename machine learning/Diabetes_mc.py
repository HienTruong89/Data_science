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
corr=data.corr()
sns.heatmap(corr,annot=True)

ax=sns.boxplot(data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

data['target']=target
q_25=data.quantile(0.25)
q_75=data.quantile(0.75)
IQR=q_75-q_25
upper=q_75+1.5*IQR
lower=q_25-1.5*IQR


data_c=data[~((data>upper)\
              |(data<lower)).any(axis=1)]

ax=sns.boxplot(data_c.drop('target',axis=1))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

## Build regression models
from sklearn.model_selection import train_test_split

X=data_c.drop('target',axis=1)
y=data_c['target']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# Simple linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

lm=LinearRegression().fit(x_train,y_train) ## need to check from here
print('Training score:',lm.score(x_train,y_train)) ## R-square
predictors=x_train.columns

coef=pd.Series(lm.coef_,predictors).sort_values()
print(coef)
y_pred=lm.predict(x_test)
pd_actual=pd.DataFrame({'actual':y_test,'predicted':y_pred})
print('Testing score:',r2_score(y_test,y_pred))
# scater plot
fig, ax=plt.subplots()
sns.scatterplot(data=pd_actual,x='actual',y='predicted')


# Partial Lineaar Regression 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,cross_val_predict

cv=KFold(n_splits=5,shuffle=True,random_state=1)
mse=[]
for i in range(1,7):
    pls=PLSRegression(n_components=i)
    score=-1*cross_val_score(pls,x_train,y_train,cv=cv,
                            scoring='neg_mean_squared_error').mean()
    mse.append(score)

plt.plot(mse)
plt.xlabel('Number of PLS Components')
plt.ylabel('MSE')
plt.title('Latent variables')

pls=PLSRegression(n_components=2)
model=pls.fit(x_train,y_train)
y_pred_cv=cross_val_predict(pls,x_train,y_train,cv=cv).reshape(-1)
mse_cv = np.sqrt(mean_squared_error(y_train, y_pred_cv))
r2_cv=r2_score(y_train,y_pred_cv) 
y_pred=model.predict(x_test).reshape(-1)
mse_pred=np.sqrt(mean_squared_error(y_test,y_pred))
r2_pred=r2_score(y_test,y_pred)
print('Results of cross validation:',r2_cv,mse_cv)
print('Results of prediction:',r2_pred,mse_pred)


res=np.array(y_pred_cv - y_train)
fig,axs=plt.subplots(ncols=2,figsize=(8,4))
axs[0].scatter(y_train,y_pred_cv) 
axs[0].plot(np.linspace(min(y_train), max(y_train), 100), np.linspace(min(y_train), max(y_train), 100), color='black', linestyle='--')
axs[0].set_xlabel("True ")
axs[0].set_ylabel("Predicted")
axs[1].scatter(y_pred_cv,res)
axs[1].axhline(y=0,color='black',linestyle='--')
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel ("Residuals")
plt.tight_layout()
plt.show()    


res=np.array(y_pred - y_test)
fig,axs=plt.subplots(ncols=2,figsize=(8,4))
axs[0].scatter(y_test,y_pred) 
axs[0].plot(np.linspace(min(y_test), max(y_test), 100), np.linspace(min(y_test), max(y_test), 100), color='black', linestyle='--')
axs[0].set_xlabel("True ")
axs[0].set_ylabel("Predicted")
axs[0].text(min(y_test), max(y_pred), f'R2 = {r2_score(y_test, y_pred):.2f}', fontsize=12)
axs[1].scatter(y_pred,res)
axs[1].axhline(y=0,color='black',linestyle='--')
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel ("Residuals")
plt.tight_layout()
plt.show()    
