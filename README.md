# Project :- Concrete-strength-prediction


## Problem Statement


To predict strength of concrete by using given features

### Importants Library

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,minmax_scale
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle

## Problem Statements 

To predict the strength of concrete by using given featurs

### Data Gathering

df=pd.read_csv("concrete_data.csv")
df

### EDA + Feature Engineering

df.info()
df.describe()

### Outliers Detection

df.head()
for col in df.columns:
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    sns.kdeplot(df[col])
    plt.title(col)

    plt.subplot(122)
    sns.boxplot(x=df[col])
    plt.title(col)

    plt.show()

#### Define Dependant And Independant Variables

x=df.drop("Strength",axis=1)
y=df[["Strength"]]

#### Model Splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

### Handling Outliers
pt=PowerTransformer()
x_train_transform=pt.fit_transform(x_train)
x_test_transform=pt.transform(x_test)
x_train_transform=pd.DataFrame(x_train_transform,columns=x_train.columns)
x_test_transform=pd.DataFrame(x_test_transform,columns=x_test.columns)

#### Check The outliers 
for col in x_train_transform.columns:
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    sns.kdeplot(x_train_transform[col])
    plt.title(col)

    plt.subplot(122)
    sns.boxplot(x=x_train_transform[col])
    plt.title(col)

    plt.show()



## 1.Linear Regression ML Algoritham

lr_transform=LinearRegression()
lr_transform.fit(x_train_transform,y_train)



### Training on scaling data ####

y_pred_train_transform=lr_transform.predict(x_train_transform)

mean_s_error=mean_squared_error(y_train,y_pred_train_transform)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train_transform)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train_transform)
print("R_squared = ",r2_s)
print("*"*50)


### Testing on scaling data ####

y_pred_test_transform=lr_transform.predict(x_test_transform)

mean_s_error=mean_squared_error(y_test,y_pred_test_transform)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test_transform)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test_transform)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train_transform =  0.81
R_squared_test_transform =  0.78


## 2.KNN Regression


#### Data Scaling
std_scale=StandardScaler()
std_scale.fit(x_train)
x_train_scaled=std_scale.transform(x_train)
x_test_scaled=std_scale.transform(x_test)
x_train_scaled=pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled=pd.DataFrame(x_test_scaled,columns=x_test.columns)
knn_reg=KNeighborsRegressor()
knn_reg.fit(x_train_scaled,y_train)


### Training on scaling data ####

y_pred_train=knn_reg.predict(x_train_scaled)

mean_s_error=mean_squared_error(y_train,y_pred_train)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train)
print("R_squared = ",r2_s)
print("*"*50)


### Testing on scaling data ####

y_pred_test=knn_reg.predict(x_test_scaled)

mean_s_error=mean_squared_error(y_test,y_pred_test)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train =  0.82
R_squared_test =  0.68


#### Find Best K values
#### when p=2

k_values = np.arange(1,15)
train_r2_score = []
test_r2_score = []
for k in k_values:
    knn_reg =  KNeighborsRegressor(n_neighbors=k, p=2)
    knn_reg.fit(x_train_scaled, y_train)
    train_r2_score.append(np.around(knn_reg.score(x_train_scaled, y_train),3))
    test_r2_score.append(np.around(knn_reg.score(x_test_scaled, y_test),3))

print("k= ",k)
print("train_r2_score list  \n",train_r2_score)
print("test_r2_score list  \n",test_r2_score)

plt.plot(k_values,train_r2_score,c="red",marker=".",ms=7,mfc="black",mec="green")
plt.plot(k_values,test_r2_score,c="blue",marker=".",ms=7,mfc="black",mec="green")
plt.grid(True)
plt.show()


#### when p=1

k_values = np.arange(1,15)
train_r2_score = []
test_r2_score = []
for k in k_values:
    knn_reg =  KNeighborsRegressor(n_neighbors=k, p=1)
    knn_reg.fit(x_train_scaled, y_train)
    train_r2_score.append(np.around(knn_reg.score(x_train_scaled, y_train),3))
    test_r2_score.append(np.around(knn_reg.score(x_test_scaled, y_test),3))

print("k= ",k)
print("train_r2_score list  \n",train_r2_score)
print("test_r2_score list  \n",test_r2_score)

plt.plot(k_values,train_r2_score,c="red",marker=".",ms=7,mfc="black",mec="green")
plt.plot(k_values,test_r2_score,c="blue",marker=".",ms=7,mfc="black",mec="green")
plt.grid(True)
plt.show()


## 3.DecisionTree Regression
dt=DecisionTreeRegressor(random_state=1)
dt.fit(x_train,y_train)


### Training ###

y_pred_train=dt.predict(x_train)

mean_s_error=mean_squared_error(y_train,y_pred_train)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train)
print("R_squared = ",r2_s)
print("*"*50)


### Testing ###

y_pred_test =dt.predict(x_test)

mean_s_error=mean_squared_error(y_test,y_pred_test)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train =  0.99
R_squared_test =  0.76


#### Hyperparameter Tuning GT
dt = DecisionTreeRegressor(random_state=1)

hyp_grid={'criterion' : ['squared_error', 'absolute_error'],
    'max_depth' : np.arange(5,10),
    'min_samples_split' : np.arange(5,15),
    'min_samples_leaf' : np.arange(3,6) }

dt_gs_cv=GridSearchCV(dt,param_grid=hyp_grid,cv=3,n_jobs=-1)
dt_gs_cv.fit(x_train,y_train)

dt_gs_cv.best_estimator_
dtc = dt_gs_cv.best_estimator_
dtc.fit(x_train,y_train)



### Training ###

y_pred_train=dtc.predict(x_train)

mean_s_error=mean_squared_error(y_train,y_pred_train)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train)
print("R_squared = ",r2_s)
print("*"*50)



### Testing ###

y_pred_test =dtc.predict(x_test)

mean_s_error=mean_squared_error(y_test,y_pred_test)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train =  0.95
R_squared_test =  0.77


## 4.RandomForest Regression

rf_reg=RandomForestRegressor(random_state=1)
rf_reg.fit(x_train,y_train)


### Training ###

y_pred_train=rf_reg.predict(x_train)

mean_s_error=mean_squared_error(y_train,y_pred_train)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train)
print("R_squared = ",r2_s)
print("*"*50)


### Testing ###

y_pred_test =rf_reg.predict(x_test)

mean_s_error=mean_squared_error(y_test,y_pred_test)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train =  0.98
R_squared_test = 0.89


#### HyperParameter Tuning
rf = RandomForestRegressor(random_state=1)

hyp_grid={'criterion' : ['squared_error', 'absolute_error'],
    'max_depth' : np.arange(5,10),
    'min_samples_split' : np.arange(5,15),
    'min_samples_leaf' : np.arange(3,6) }

rf_gs_cv=GridSearchCV(rf,param_grid=hyp_grid,cv=3,n_jobs=-1)
rf_gs_cv.fit(x_train,y_train)
rf_gs_cv.best_estimator_
rfc=rf_gs_cv.best_estimator_
rfc.fit(x_train,y_train)


### Training ###

y_pred_train=rfc.predict(x_train)

mean_s_error=mean_squared_error(y_train,y_pred_train)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train)
print("R_squared = ",r2_s)
print("*"*50)


### Testing ###

y_pred_test =rfc.predict(x_test)

mean_s_error=mean_squared_error(y_test,y_pred_test)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train = 0.95
R_squared_test = 0.87


## 5.AdaboostRegression
ad_reg=AdaBoostRegressor(random_state=1)
ad_reg.fit(x_train,y_train)


### Training ###

y_pred_train=ad_reg.predict(x_train)

mean_s_error=mean_squared_error(y_train,y_pred_train)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train)
print("R_squared = ",r2_s)
print("*"*50)

### Testing ###

y_pred_test =ad_reg.predict(x_test)

mean_s_error=mean_squared_error(y_test,y_pred_test)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train = 0.824
R_squared_test = 0.726


#### HyperParameter Tuning
ad_reg =AdaBoostRegressor(random_state=1)

hyp_grid={'n_estimators' : np.arange(10,100,4),
    'learning_rate' : np.arange(0,1,0.1) }

ad_reg_gs_cv=GridSearchCV(ad_reg,param_grid=hyp_grid,cv=3,n_jobs=-1)
ad_reg_gs_cv.fit(x_train,y_train)
ad_reg_gs_cv.best_estimator_
ad_regc=ad_reg_gs_cv.best_estimator_
ad_regc.fit(x_train,y_train)


### Training ###

y_pred_train=ad_regc.predict(x_train)

mean_s_error=mean_squared_error(y_train,y_pred_train)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_train,y_pred_train)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_train,y_pred_train)
print("R_squared = ",r2_s)
print("*"*50)


### Testing ###

y_pred_test =ad_regc.predict(x_test)

mean_s_error=mean_squared_error(y_test,y_pred_test)
print("MSE = ",mean_s_error)
print("*"*50)
mean_a_error=mean_absolute_error(y_test,y_pred_test)
print("MAE = ",mean_a_error)
print("*"*50)
r2_s=r2_score(y_test,y_pred_test)
print("R_squared = ",r2_s)
print("*"*50)
R_squared_train = 0.82
R_squared_test = 0.732


### Conclusion

     ML Models    -     Linear Reg.       KNN.Re         Decision Tree.Reg        Random Forest.Reg            Adaboost.Reg              
Training Accuracy -        81.00      82.00   82.00        1.00   0.87             0.98    0.95(H)         0.82    0.82(H) 
Testing Accuracy  -        78.00      68.00   68.00        0.71   0.77             0.89    0.87(H)         0.71    0.73(H)





### Best Model
1.Random Forest.Reg
2.Linear Regression

### Best Model Test File

with open("Concrete....RandomForest_reg_model.pkl", 'wb') as f:
    pickle.dump(rfc, f)