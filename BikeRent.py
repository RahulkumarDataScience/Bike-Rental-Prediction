# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:30:11 2019

@author: Rahhy
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from collections import Counter
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#load dataset
train = pd.read_csv("day.csv")

#Removing redundant columns like index, datetime, casual , registered
train = train.drop(['instant', 'dteday', 'casual', 'registered'], axis =1)

#Renaming the columns of the dataset
train= train.rename(columns={'yr':'year','mnth':'month','weathersit':'weather','temp':'temperature', 'hum':'humidity','cnt':'count'})
    
#Check renamed columns
print(train.columns)

############################ Missing Value Analysis #############################################################

#Create dataframe with missing percentage toc check missing values in our both dataset
def missin_val(df):
    missin_val = pd.DataFrame(df.isnull().sum())
    missin_val = missin_val.reset_index()
    missin_val = missin_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
    missin_val['Missing_percentage'] = (missin_val['Missing_percentage']/len(df))*100
    missin_val = missin_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
    return(missin_val)

print("The missing value percentage in training data : \n\n",missin_val(train))
print("\n")

##################################################  Outlier Analysis ###############################################################################################

train.plot(kind='box', subplots=True, layout=(8,3), sharex=False, sharey=False, fontsize=8)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top= 3,wspace=0.2, hspace=0.2)
plt.show()

#seperate continuous and categorical variables-
#continuous variable-
cnames= ['temperature', 'atemp', 'humidity', 'windspeed', 'count']

#categorical variables-
cat_cnames=['season', 'year', 'month', 'holiday', 'weekday', 'workingday','weather']

##Detect and replace outliers as NA from data
for i in cnames:
    print(i)
    q75, q25 = np.percentile(train.loc[:,i], [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)
    
    train.loc[train[i] < min,i] = np.nan
    train.loc[train[i] > max,i] = np.nan

#check NA in data
print(train.isnull().sum())

#Variables humidity and windspeed are having NA's
#impute NA with median
train['humidity']= train['humidity'].fillna(train['humidity'].median())
train['windspeed']= train['windspeed'].fillna(train['windspeed'].median())

#check NA in data
print(train.isnull().sum())

######################################### Data Exploration #########################################################################################################

#Histogram Plot of season Column
plt.figure(figsize=(7,7))
plt.hist(train['season'],bins=6)
plt.xlabel('season')
plt.ylabel('Frequency')

#Histogram Plot of year Column
plt.figure(figsize=(7,7))
plt.hist(train['year'],bins=5)
plt.xlabel('year')
plt.ylabel('Frequency')

#Histogram Plot of month Column
plt.figure(figsize=(7,7))
plt.hist(train['month'],bins=12)
plt.xlabel('month')
plt.ylabel('Frequency')

#Histogram Plot of weekday Column
plt.figure(figsize=(7,7))
plt.hist(train['weekday'],bins=10)
plt.xlabel('Different Days of the week')
plt.ylabel('Frequency')

#Histogram Plot of workingday Column
plt.figure(figsize=(7,7))
plt.hist(train['workingday'],bins=10)
plt.xlabel('workingday')
plt.ylabel('Frequency')

#Histogram Plot of temperature Column
plt.figure(figsize=(7,7))
plt.hist(train['temperature'],bins=10)
plt.xlabel('temperature')
plt.ylabel('Frequency')

#Histogram Plot of atemp Column
plt.figure(figsize=(7,7))
plt.hist(train['atemp'],bins=10)
plt.xlabel('atemp')
plt.ylabel('Frequency')

#Histogram Plot of humidity Column
plt.figure(figsize=(7,7))
plt.hist(train['humidity'],bins=10)
plt.xlabel('humidity')
plt.ylabel('Frequency')

#Histogram Plot of windspeed Column
plt.figure(figsize=(7,7))
plt.hist(train['windspeed'],bins=10)
plt.xlabel('windspeed')
plt.ylabel('Frequency')

#Histogram Plot of count Column
plt.figure(figsize=(7,7))
plt.hist(train['count'],bins=10)
plt.xlabel('count')
plt.ylabel('Frequency')

################################################## Bivariate Plots #################################################################################################

for i in cat_cnames:
    sns.catplot(x=i,y="count",data=train)
    fname = str(i)+'.pdf'

##################################################  Density Plots ##################################################################################################

sns.kdeplot(train['season'], shade = True)
sns.kdeplot(train['month'], shade = True)
sns.kdeplot(train['holiday'], shade = True)
sns.kdeplot(train['weekday'], shade = True)
sns.kdeplot(train['workingday'], shade = True)
sns.kdeplot(train['weather'], shade = True)
sns.kdeplot(train['temperature'], shade = True)
sns.kdeplot(train['atemp'], shade = True)
sns.kdeplot(train['humidity'], shade = True)
sns.kdeplot(train['windspeed'], shade = True)
sns.kdeplot(train['count'], shade = True)

# Feature Selection
##Correlation analysis

#extract only numeric variables in dataframe for correlation-
df_corr= train.loc[:,cnames]

#generate correlation matrix-
corr_matrix= df_corr.corr()
(print(corr_matrix))
    
f,ax= plt.subplots(figsize=(8,8))
sns.heatmap(corr_matrix,mask=np.zeros_like(corr_matrix,dtype=np.bool),cmap=sns.diverging_palette(240,120,as_cmap=True),
           square=True,ax=ax,annot=True)
plt.title("Correlation Plot")

X_train = train.loc[:,train.columns != 'count']
y_train = train['count']

pca = PCA(n_components=10)
pca.fit(X_train)
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.show()

pca = PCA(n_components=5)
X = pca.fit(X_train).transform(X_train)


###### Sampling the splits through stratified way ###########
X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2)

###### KNN Modelling ########
def train_KNN(n_neigh):
    knn = KNeighborsRegressor(n_neighbors= n_neigh)
    knn_model = knn.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print('n_neighbours : {}  ----KNN rmse: {}'.format(n_neigh,(sqrt(mean_squared_error(y_test,y_pred)))))

for n_neigh in [10,20,30,40,50,60,70,80,90,100]:
    train_KNN(n_neigh)

KNN_model = KNeighborsRegressor(n_neighbors= 10).fit(X_train , y_train)
KNN_pred_train = KNN_model.predict(X_train)
KNN_pred= KNN_model.predict(X_test)
KNN_pred_test = KNN_model.predict(X_test)
print("Train Data")
print('n_neighbours : {}  ----KNN rmse: {}'.format(n_neigh,  (sqrt(mean_squared_error(y_train,KNN_pred_train)))))
print("Test Data")
print('n_neighbours : {}  ----KNN rmse: {}'.format(n_neigh,  (sqrt(mean_squared_error(y_test,KNN_pred)))))
print("Accuracy : ")
KNN_model.score(X_train, y_train)

####### Linear Regression ######
ols = LinearRegression()
ols_model = ols.fit(X_train, y_train)
y_pred_train = ols_model.predict(X_train)
y_pred = ols_model.predict(X_test)

print("Train Data")
print('Ordinary Least Squares rmse: {}'.format(sqrt(mean_squared_error(y_train,y_pred_train))))
print("Test Data")
print('Ordinary Least Squares rmse: {}'.format(sqrt(mean_squared_error(y_test,y_pred))))
print("Accuracy : ")
ols_model.score(X_train, y_train)

#######Ridge Regression ######
def train_ridge(alpha):
    ridge = Ridge(alpha= alpha)
    ridge_model = ridge.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    print('alpha : {}  ----Ridge rmse: {}'.format(alpha, (sqrt(mean_squared_error(y_test,y_pred)))))

for alpha in [0.1,0.5, 1.0,3.0,7.0,10.0]:
    train_ridge(alpha)
    
ridge_model = Ridge(alpha= 0.1).fit(X_train , y_train)
ridge_pred_train = ridge_model.predict(X_train)
ridge_pred= ridge_model.predict(X_test)
print("Train Data")
print('Ridge rmse: {}'.format(sqrt(mean_squared_error(y_train,ridge_pred_train))))
print("Test Data")
print('Ridge rmse: {}'.format(sqrt(mean_squared_error(y_test,ridge_pred))))
print("Accuracy : ")
ridge_model.score(X_train, y_train)

####### Lasso Regression ######
def train_lasso(alpha):
    lasso = Lasso(alpha= alpha)
    lasso_model = lasso.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    print('alpha : {}  ---- Lasso rmse: {}'.format(alpha,(sqrt(mean_squared_error(y_test,y_pred)))))

for alpha in [0.1,0.5, 1.0,3.0,7.0,10.0]:
    train_lasso(alpha)

lasso_model = Lasso(alpha= 0.1).fit(X_train , y_train)
lasso_pred_train = lasso_model.predict(X_train)
lasso_pred= lasso_model.predict(X_test)
print("Train Data")
print('Lasso rmse: {}'.format(sqrt(mean_squared_error(y_train,lasso_pred_train))))
print("Test Data")
print('Lasso rmse: {}'.format(sqrt(mean_squared_error(y_test,lasso_pred))))
print("Accuracy :")
lasso_model.score(X_train, y_train)

###### SVM Modelling ##########
def train_SVR(C, gamma):
    svr = SVR(C= C, gamma = gamma)
    svr_model = svr.fit(X_train, y_train)
    y_pred = svr_model.predict(X_test)
    print('C : {} , gamma : {} ----SVR rmse: {}'.format(C, gamma ,(sqrt(mean_squared_error(y_test,y_pred)))))
    
for C in [1, 10, 100,1000]:
    for gamma in [0.001, 0.0001]:
        train_SVR(C, gamma)
        
svr_model = SVR(C= 1, gamma = 0.001).fit(X_train , y_train)
svr_pred_train = svr_model.predict(X_train)
svr_pred= svr_model.predict(X_test)
print("Train Data")
print('Support Vector Regression rmse: {}'.format(sqrt(mean_squared_error(y_train,svr_pred_train))))
print("Test Data")
print('Support Vector Regression rmse: {}'.format(sqrt(mean_squared_error(y_test,svr_pred))))
print("Accuracy :")
svr_model.score(X_train, y_train)

###### DecisionTree Modelling ##########
def DT(depth):
    dt = tree.DecisionTreeRegressor( max_depth = depth)
    dt_model = dt.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    print('depth : {} ----  Decision Tree rmse: {}'.format(depth,(sqrt(mean_squared_error(y_test,y_pred)))))
    
for depth in [1,2,5,10,20]:
    DT(depth)
    
dt_model = tree.DecisionTreeRegressor(max_depth =20).fit(X_train, y_train)
dt_pred_train = dt_model.predict(X_train)
dt_pred= dt_model.predict(X_test)
print("Train Data")
print('Decision Tree rmse: {}'.format(sqrt(mean_squared_error(y_train,dt_pred_train))))
print("Test Data")
print('Decision Tree rmse: {}'.format(sqrt(mean_squared_error(y_test,dt_pred))))
dt_model.score(X_train, y_train)

###### GBR Modelling ##########
def GBR(depth, learning_rate):
    gbr = GradientBoostingRegressor( max_depth = depth, learning_rate =learning_rate)
    gbr_model = gbr.fit(X_train, y_train)
    y_pred = gbr_model.predict(X_test)
    print('depth : {}, learning_rate{}  ---- Gradient Boosting Regression rmse: {}'.format(depth, learning_rate, (sqrt(mean_squared_error(y_test,y_pred)))))  
    
for depth in [1,2,5]:
    for learning_rate in [0.001,0.01,0.1]:
        GBR(depth, learning_rate)
        
gbr_model = GradientBoostingRegressor(max_depth= 5,learning_rate = 0.1).fit(X_train , y_train)
gbr_pred_train = gbr_model.predict(X_train)
gbr_pred= gbr_model.predict(X_test)
print("Train Data")
print('GBDT rmse: {}'.format(sqrt(mean_squared_error(y_train,dt_pred_train))))
print("Test Data")
print('GBDT rmse: {}'.format(sqrt(mean_squared_error(y_test,dt_pred))))
print("Accuracy :")
gbr_model.score(X_train, y_train)

###### RandomForest Modelling ##########
def train_RF(n_est, depth):
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, n_jobs=-1)
    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print('depth : {}, n_estimators : {}  ---- Random Forest Regression rmse: {}'.format(depth, n_est, (sqrt(mean_squared_error(y_test,y_pred)))))  

for n_est in [100, 200]:
    for depth in [2, 5, 10 , 20, 30]:
        train_RF(n_est, depth)
        
rf_model = RandomForestRegressor(max_depth= 30, n_estimators = 200).fit(X_train , y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred= rf_model.predict(X_test)
print("Train Data")
print('Random Forest rmse: {}'.format(sqrt(mean_squared_error(y_train,rf_pred_train))))
print("Test Data")
print('Random Forest rmse: {}'.format(sqrt(mean_squared_error(y_test,rf_pred))))
print("Accuracy :")
rf_model.score(X_train, y_train)



























    


