import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics  import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense,Input, Activation, BatchNormalization, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

K.image_data_format()



## Definitions 
def SNV(X):
    mean_t = X.mean(axis=1)
    std_t = X.std(axis=1)
    SNV= (X - mean_t.reshape((-1,1)))/std_t.reshape((-1,1))
    return SNV

def SG1(X):
    SG1= savgol_filter(X, window_length=3, polyorder=2, deriv=1)
    return SG1

def Norm(X):
    Norm=(X - X.min()) / (X.max() - X.min())
    return Norm
# Export data
os.chdir(r'C:\Users\Hien Thi Dieu Truong\OneDrive\Research work\Deep learning article\Improve CNN model')

df=pd.read_csv('spectral-data-cal.csv')
df.isnull().sum()
# Detection of outliers
subset=df.loc[:,['UMF','MPI']]
sns.boxplot(data=subset)
# plot the spectra
Data=np.array(df.drop(['UMF','MPI'],axis=1)).astype('float32')
wavelengths=np.array(list(df.drop(['UMF','MPI'],axis=1).columns)).astype('float32')
w=np.round(wavelengths,0).astype('int')

fig, ax=plt.subplots(figsize=(10,8))
for i in range(Data.shape[0]):
    plt.plot(w,Data[i,:])
ax.set_xlabel('Wavelengths',fontsize=15)
ax.set_ylabel('Absorption intensity',fontsize=15)
plt.show()
# Pre-process data
Data_pre=SNV(Data)
fig, ax=plt.subplots(figsize=(10,8))
for i in range(Data_pre.shape[0]):
    plt.plot(w,Data_pre[i,:])
ax.set_xlabel('Wavelengths',fontsize=15)
ax.set_ylabel('Processing data',fontsize=15)
plt.show()

# Define data 
X=Data_pre[:,:]  
# Define classes (0: mono-manuka,1: multi-manuka, 2: non-manuka)
classes=df['MPI']
classes_MPI = np.unique(classes)
class_0= np.where(classes_MPI==0)[0]
class_1 = np.where(classes_MPI==1)[0]
class_2 = np.where(classes_MPI==2)[0]
######### Use data from PLSDA/SVC ###########
# No spliting is applied
y_train = classes
x_train = X[:,:]
# Use one-hot encoding "to_categorical" for caterogizing classes
y_train_cat = tensorflow.keras.utils.to_categorical(y_train)
# Expand the x-train to the final axis 
x_train_model = np.expand_dims(x_train, axis=-1)
##############################
# test data 
df_test=pd.read_csv('spectral-data-val.csv')
df_test_classes = np.array(list(df_test.iloc[:]['MPI'])).reshape(-1,1).astype('float32')
x_test = np.array(df_test.drop(['UMF','MPI'], axis=1)).astype('float32')

# Pre-process data
x_test=SNV(x_test)
fig, ax=plt.subplots(figsize=(10,8))
for i in range(x_test.shape[0]):
    plt.plot(w,x_test[i,:])
ax.set_xlabel('Wavelengths',fontsize=15)
ax.set_ylabel('Processing data',fontsize=15)
plt.show()


classes_test= np.unique(df_test_classes)
C0 = np.where(df_test_classes ==0)[0]
C1 = np.where(df_test_classes==1)[0]
C2 = np.where(df_test_classes ==2)[0]
y_test=df_test_classes[:,:]
y_test_cat=tensorflow.keras.utils.to_categorical(y_test)
y_test=y_test_cat
########## Expand the data #########
x_test = np.expand_dims(x_test, axis=-1)
##### CNN model #####
def get_model(fil_1=8, fil_2=16, fil_3=32, unit_1=128, unit_2=8, drop_out=0.0):
    
    model = Sequential()    
    model.add(Conv1D(fil_1, kernel_size=3, input_shape= (234,1), strides=1, padding='valid', kernel_regularizer=regularizers.l2(1e-04),drop_out=drop_out))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    
    model.add(Conv1D(fil_2, kernel_size=3, strides=1, padding='valid', kernel_regularizer=regularizers.l2(1e-04),drop_out=drop_out))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    
    model.add(Conv1D(fil_3, kernel_size=3, strides=1, padding='valid', kernel_regularizer=regularizers.l2(1e-04),drop_out=drop_out))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(unit_1, activation = 'relu', kernel_regularizer=regularizers.l2(1e-04),drop_out=drop_out))
    model.add(Dense(unit_2, activation = 'relu', kernel_regularizer=regularizers.l2(1e-04),drop_out=drop_out))
    model.add(Dense(3, activation = 'softmax'))
    
    adam     = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)    
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    
    return model

get_model.summary()
##########################
dataset_path = r"C:\Users\Hien Thi Dieu Truong\OneDrive\Research work\Deep learning article\Improve CNN model\results"
os.chdir(dataset_path)

acc_cal = []
acc_val = []

f_avg_cal = []
f_avg_val = []

p_cal = []
p_val = []

r_cal = []
r_val = []
#### grid search 
param_grid={'fil_1':[8,16,32],'fil_2':[16,32,64],
             'fil_3':[32,64,128],'unit_1':[128,256,512],
             'unit_2':[8,16,64],'dropout_rate':[0.0,0.1,0.2]}
# run model with grid search 
cnn_model = tf.Keras.Classifier(build_fn=get_model, epochs=200, batch_size=16, verbose=0)
grid = GridSearchCV(estimator=cnn_model, param_grid=param_grid, cv=KFold(n_splits=5))
grid_result = grid.fit(x_train_model, y_train_cat)

print(f'Best parameters: {grid_result.best_params_}')
print(f'Best score: {grid_result.best_score_}')

# Run model with optimised parameters
from sklearn.model_selection import KFold
n_fold = 5
kf=KFold(n_splits=n_fold,random_state=101,shuffle=True)
k=1

for fold_index,train_index,test_index in kf.split(x_train_model,y_train_cat):

    x_cal = x_train_model[train_index]
    y_cal = y_train_cat[train_index]
    x_val = x_train_model[test_index]
    y_val = y_train_cat[test_index]

    y_true_cal = np.argmax(y_cal,axis=1)
    y_true_val = np.argmax(y_val,axis=1)
    
    modelfilepath='model_best_'+str(k)+'.hdf5'
    k=k+1
    checkpointer = ModelCheckpoint(modelfilepath,
                               monitor = "val_loss",
                               verbose=0, save_best_only=True)
    model_best = get_model()
    
    hist=model_best.fit(x_cal, y_cal,validation_data = (x_val, y_val),
                 batch_size=16, epochs=200, verbose=1, callbacks=[checkpointer])
    
    model_best = get_model()
    model_best.load_weights(modelfilepath)
    
    y_pred_cal = np.argmax(model_best.predict(x_cal),axis=1)
    y_pred_val = np.argmax(model_best.predict(x_val),axis=1)
    
    acc_cal.append(accuracy_score(y_true_cal,y_pred_cal))
    acc_val.append(accuracy_score(y_true_val, y_pred_val))
    
    f_avg_cal.append(f1_score(y_true_cal, y_pred_cal, average="macro"))
    f_avg_val.append(f1_score(y_true_val, y_pred_val, average="macro"))
    
    p_cal.append(precision_score(y_true_cal, y_pred_cal, average=None, zero_division=0))
    p_val.append(precision_score(y_true_val, y_pred_val, average=None, zero_division=0))
    
    r_cal.append(recall_score(y_true_cal, y_pred_cal, average=None, zero_division=0))
    r_val.append(recall_score(y_true_val, y_pred_val, average=None, zero_division=0))    
 
 
dict1 = {'acc_cal': acc_cal, 'acc_val': acc_val, 'f_avg_cal': f_avg_cal, 'f_avg_val': f_avg_val} 
results = pd.DataFrame(dict1) 
print('Classification results:',results)
results.to_csv('model_acc_favg.csv')

##################################
print(hist.history.keys())
# summarize history for loss
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(train_loss, 'r--')
plt.plot(val_loss, 'b-')
plt.legend(['train loss', 'Val loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss-entropy')
plt.show()


# test the model with external dataset       
modelfilepath='model_best_1.hdf5'
model_test = get_model()
model_test.load_weights(modelfilepath)
####### checking the final model #########
y_true, y_pred = np.argmax(y_train,axis=1), np.argmax(model_test.predict(x_train),axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
####### validation #########
y_true, y_pred = np.argmax(y_val,axis=1), np.argmax(model_test.predict(x_val),axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
####### test ############
y_true_test, y_pred_test = np.argmax(y_test,axis=1), np.argmax(model_test.predict(x_test),axis=1)
    
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))


