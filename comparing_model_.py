#Importing necessary libraries 
from math import gamma
from tabnanny import verbose
from catboost import cv
import pandas as pd
import numpy as np
from scipy import rand
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from skompiler import skompile 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


#Importing and copy dataset
print('Sonar data importing...')

sonardata = pd.read_csv('Copy of sonar data.csv',header=None)
df = sonardata.copy()

print('Sonar Data imported.')
#Data Preprocessing // Preparing for models
print('Data Preprocessing...')

df.rename({df.columns[60]:'Type'},axis=1,inplace=True)
dummies = pd.get_dummies(df['Type'])
mine_dummy = dummies['M']
x = df.drop(df.columns[60],axis=1)
cleaned_df = pd.concat([x,mine_dummy],axis=1)
cleaned_df = cleaned_df.rename({'M':'is_mine'},axis=1)
y = cleaned_df['is_mine']

print('Preprocessing completed.')
#Test-train split
print('Spliting..')

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=42)

print(f'\n***********************\nX Train Shape : {xtrain.shape}\nX Test Shape : {xtest.shape}\nY Train Shape : {ytrain.shape}\nY Test Shape : {ytest.shape}\n\nSplit completed.\n\n')


def apply_logistic_regression():

    model = LogisticRegression(solver='liblinear').fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    kcv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    cv_score = cross_val_score(
        model,
        xtest,
        ytest,
        cv=kcv
    ).mean()

    Scores = [acc1,cv_score,'Logistic']
    print('\n\nLogistic Regression completed.\n\n')
    return Scores


def apply_knn():

    model = KNeighborsClassifier().fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    model_1 = KNeighborsClassifier()
    params = {'n_neighbors':np.arange(1,50)}
    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        n_jobs=-1,
        verbose=2
    )
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = KNeighborsClassifier(
        n_neighbors=cvmodel.best_params_['n_neighbors']
    ).fit(xtrain,ytrain)

    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))
    Scores = [acc1,acc2,'KNN']
    print('\n\nKNN Classifier completed.\n\n')
    return Scores


def apply_cart_classifier():
    model = DecisionTreeClassifier().fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    params = {
        'max_depth':np.arange(1,11),
        'min_samples_split':np.arange(2,51)
    }
    model_1 = DecisionTreeClassifier()
    cvmodel = GridSearchCV(model_1,params,cv=10,n_jobs=-1,verbose=2)
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = DecisionTreeClassifier(
        max_depth=cvmodel.best_params_['max_depth'],
        min_samples_split=cvmodel.best_params_['min_samples_split']
    ).fit(xtrain,ytrain)
    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))
    Scores = [acc1,acc2,'CART']

    cart_python_code = skompile(tunedmodel.predict).to('python/code')
    Scores.append(cart_python_code)
    print('\n\nCART Classifier completed.\n\n')
    return Scores


def apply_naivebayes():
    model = GaussianNB().fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    kcv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )
    acc2 = cross_val_score(
        model,
        xtest,
        ytest,
        cv=kcv
    ).mean()

    Scores = [acc1,acc2,'NaiveB']
    print('\n\nNaive Bayes completed.\n\n')
    return Scores


def apply_svm():
    model = SVC(kernel='linear').fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    model_1 = SVC(kernel='linear')
    params = {
        'C':np.arange(1,11)
    }
    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        n_jobs=-1,
        verbose=2
    )
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = SVC(
        kernel='linear',
        C=cvmodel.best_params_['C']
    ).fit(xtrain,ytrain)
    
    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))

    Scores = [acc1,acc2,'SVC']
    print('\n\nSVC completed.\n\n')
    return Scores


def apply_gbm():
    model = GradientBoostingClassifier().fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    model_1 = GradientBoostingClassifier()
    params = {
        'learning_rate':[0.001,0.01,0.1,0.2],
        'max_depth':[3,5,8,10,50,100],
        'n_estimators':[200,500,1000],
        'subsample':[1,0.5,0.75],
    }

    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        verbose=2,
        n_jobs=-1
    )
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = GradientBoostingClassifier(
        learning_rate=cvmodel.best_params_['learning_rate'],
        max_depth=cvmodel.best_params_['max_depth'],
        n_estimators=cvmodel.best_params_['n_estimators'],
        subsample=cvmodel.best_params_['subsample']
    ).fit(xtrain,ytrain)

    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))

    Scores = [acc1,acc2,'GBM']
    print('\n\nGBM Classifier completed.\n\n')
    return Scores

def apply_rf():
    model = RandomForestClassifier().fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    params = {
        'max_depth':[3,5,8,10,50,100],
        'n_estimators':[200,500,1000],
        'max_features':[2,5,8],
        'min_samples_split':[2,3,5,8]
    }
    model_1 = RandomForestClassifier()
    
    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        n_jobs=-1,
        verbose=2
    )
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = RandomForestClassifier(
        max_depth=cvmodel.best_params_['max_depth'],
        min_samples_split=cvmodel.best_params_['min_samples_split'],
        n_estimators=cvmodel.best_params_['n_estimators'],
        max_features=cvmodel.best_params_['max_features']
    ).fit(xtrain,ytrain)
    
    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))
    Scores = [acc1,acc2,'RF']
    print('\n\nRandom Forest Classifier completed.\n\n')
    return Scores

def apply_rbf_svm():
    model = SVC(kernel='rbf').fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    model_1 = SVC(kernel='rbf')
    params = {
        'C':[0.0001,0.001,0.01,0.1,1,5,10,50,100],
        'gamma':[0.0001,0.001,0.01,0.1,1,5,10,50,100]
    }
    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        n_jobs=-1,
        verbose=2
    )
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = SVC(
        kernel='rbf',
        C=cvmodel.best_params_['C'],
        gamma=cvmodel.best_params_['gamma']
    ).fit(xtrain,ytrain)

    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))
    Scores = [acc1,acc2,'RBF']
    print('\n\nRBF SVC completed.\n\n')
    return Scores

def apply_neural_networks():

    #Preprocessing for test and train variables
    scaler = StandardScaler()
    scaler.fit(xtrain)
    xtrainscaled = scaler.transform(xtrain)
    xtestscaled = scaler.transform(xtest)
    ###########################################

    model = MLPClassifier().fit(xtrainscaled,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtestscaled))

    params = {
        'hidden_layer_sizes':[(10,10,10),
                            (100,100,100),
                            (100,100),
                            (3,5),
                            (5,3)],
        'activation':['logistic','relu'],
        'alpha':[0.1,0.01,0.02,0.005,0.0001,0.0000005],
        'solver':['lbfgs','adam','sgd']
    }
    model_1 = MLPClassifier()
    
    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        n_jobs=-1,
        verbose=2
    )
    cvmodel.fit(xtrainscaled,ytrain)

    tunedmodel = MLPClassifier(
        hidden_layer_sizes=cvmodel.best_params_['hidden_layer_sizes'],
        activation=cvmodel.best_params_['activation'],
        solver=cvmodel.best_params_['solver'],
        alpha=cvmodel.best_params_['alpha']
    ).fit(xtrain,ytrain)

    acc2 = accuracy_score(ytest,tunedmodel.predict(xtestscaled))
    Scores = [acc1,acc2,'Neural']
    print('\n\nNeural Network Classifier completed.\n\n')
    return Scores


def apply_xgb():
    model = XGBClassifier().fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    model_1 = XGBClassifier()
    params = {
        'colsample_bytree':[0.4,0.5,0.6],
        'n_estimators':[100,200,500,1000],
        'max_depth':[2,3,4,5,6],
        'learning_rate':[0.1,0.01,0.05]
    }
    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        n_jobs=-1,
        verbose=2
    )
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = XGBClassifier(
        colsample_bytree = cvmodel.best_params_['colsample_bytree'],
        learning_rate = cvmodel.best_params_['learning_rate'],
        max_depth = cvmodel.best_params_['max_depth'],
        n_estimators = cvmodel.best_params_['n_estimators']
    ).fit(xtrain,ytrain)

    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))
    Scores = [acc1,acc2,'XGB']
    print('\n\nXGB Classifier completed.\n\n')
    return Scores


def apply_lgb():
    model = LGBMClassifier().fit(xtrain,ytrain)
    acc1 = accuracy_score(ytest,model.predict(xtest))

    params = {
        'max_depth':[3,4,5,6],
        'learning_rate':[0.1,0.01,0.02,0.05],
        'n_estimators':[100,500,1000,2000],
        'min_child_samples':[2,5,10],
        'subsample':[0.6,0.8,1.0]
    }
    model_1 = LGBMClassifier()

    cvmodel = GridSearchCV(
        model_1,
        params,
        cv=10,
        n_jobs=-1,
        verbose=2
    )
    cvmodel.fit(xtrain,ytrain)

    tunedmodel = LGBMClassifier(
        max_depth=cvmodel.best_params_['max_depth'],
        learning_rate=cvmodel.best_params_['learning_rate'],
        n_estimators=cvmodel.best_params_['n_estimators'],
        min_child_samples=cvmodel.best_params_['min_child_samples'],
        subsample=cvmodel.best_params_['subsample']
    ).fit(xtrain,ytrain)

    acc2 = accuracy_score(ytest,tunedmodel.predict(xtest))
    Scores = [acc1,acc2,'LGBM']
    print('\n\nLGBM Classifier completed.\n\n')
    return Scores


logistic = apply_logistic_regression()
knn = apply_knn()
neural = apply_neural_networks()
gbm = apply_gbm()
lgbm = apply_lgb()
xgbm = apply_xgb()
rbf = apply_rbf_svm()
svm = apply_svm()
naiveB = apply_naivebayes()
rf = apply_rf()
cart = apply_cart_classifier()

G = [logistic,knn,neural,gbm,lgbm,xgbm,rbf,svm,naiveB,rf,cart]

columns = ['Unvalid Test Score','Tuned and Validated Test Score']
unv_test = []
val_test = []
indexes = []

for i in G:
    unv_test.append(i[0])
    val_test.append(i[1])
    indexes.append(i[2])

overview = pd.DataFrame({
    columns[0]:unv_test,
    columns[1]:val_test
},index=indexes)

overview.sort_values(by='Tuned and Validated Test Score',axis=0).plot(kind='bar')
plt.show()
