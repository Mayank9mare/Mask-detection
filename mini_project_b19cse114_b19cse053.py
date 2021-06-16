# -*- coding: utf-8 -*-
"""mini_project_b19cse114_b19cse053.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hCdZSnanKFuG2-M47M-D05jE4Bmb2vGF
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
rcParams['figure.figsize']= 5,5
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_curve,auc

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv("/content/drive/MyDrive/bonus_dataset/data_64.csv")
df.head()

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x.head()

for i in range(1,6):
  plt.figure(figsize=(5,5))
  plt.imshow(np.array(x)[i].reshape(64,64,3))
  plt.show()
  plt.figure(figsize=(5,5))
  plt.imshow(np.array(x)[-i].reshape(64,64,3))
  plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True ,random_state=42)

print(np.unique(y_train , return_counts=True))
print(np.unique(y_test,return_counts=True))

#PCA conversion
from sklearn.decomposition import PCA
pca = PCA(0.95)
x_trn = pca.fit_transform(x_train)
x_tst = pca.transform(x_test)
y_trn = y_train
y_tst = y_test

def give_roc(model,x_test,y_test):
  a = model.predict(x_test)
  fpr , tpr , _ = roc_curve(y_test,a)
  plt.plot(fpr,tpr,label = f'auc = {auc(fpr,tpr)}')
  plt.legend()
  plt.show()

from sklearn.neural_network import MLPClassifier

clf1  = MLPClassifier()
clf1.fit(x_trn,y_trn)

print("Accuracy",clf1.score(x_tst,y_tst))
print("confusion matrix is \n",confusion_matrix(y_tst,clf1.predict(x_tst)))

give_roc(clf1,x_tst,y_tst)

cv_score = cross_val_score(estimator=MLPClassifier(),X=x_trn,y=y_trn,cv=5)
cv_score

plt.figure(figsize=(5,5))
sns.boxplot(cv_score)

from sklearn.neighbors import KNeighborsClassifier

clf2=KNeighborsClassifier(n_neighbors=12)
clf2.fit(x_trn,y_trn)

print("Accuracy",clf2.score(x_tst,y_tst))
print("confusion matrix is \n",confusion_matrix(y_tst,clf2.predict(x_tst)))

give_roc(clf2,x_tst,y_tst)

estimators ={i:KNeighborsClassifier(n_neighbors=i) for i in range(5,50,5)}
estimators

accuracy_list = []
for C,model in estimators.items():
  model.fit(x_trn,y_trn)
  accuracy_list.append(model.score(x_tst,y_tst))

plt.figure(figsize=(5,5))
plt.plot(list(estimators.keys()),accuracy_list)
plt.xlabel('C')
plt.ylabel('accuracy')

cv_score4 = cross_val_score(estimator=KNeighborsClassifier(n_neighbors=12),X=x_trn,y=y_trn,cv=5)
cv_score4

sns.boxplot(cv_score4)



x_tst.shape

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

svm_clf=Pipeline([
    ("standardscaler",StandardScaler()),
    ('svc',SVC(gamma="auto",probability=True,kernel="rbf",C=10,decision_function_shape='ovo'))])

svm_clf.fit(x_trn,y_trn)

print('Accuracy',svm_clf.score(x_tst,y_tst))
print("confusion matrix is \n",confusion_matrix(y_tst,svm_clf.predict(x_tst)))

give_roc(svm_clf,x_tst,y_tst)

cv_score1 = cross_val_score(estimator=svm_clf,X=x_trn,y=y_trn,cv=5)
cv_score1

sns.boxplot(cv_score1)

estimators ={i:Pipeline([("standardscaler",StandardScaler()),('svc',SVC(gamma="auto",kernel="rbf",C=i,decision_function_shape='ovo'))]) for i in range(2,50,5)} #{'2':SVC(C=2),'3':SVC(C=3)}
estimators

from sklearn.metrics import accuracy_score
accuracy_list = []
for C,model in estimators.items():
  model.fit(x_trn,y_trn)
  a = model.predict(x_tst)
  accuracy_list.append(accuracy_score(a,y_tst))

plt.figure(figsize=(5,5))
plt.plot(list(estimators.keys()),accuracy_list)
plt.xlabel('C')
plt.ylabel('accuracy')

from sklearn.ensemble import RandomForestClassifier

clfR= RandomForestClassifier(n_estimators=190)
clfR.fit(x_trn,y_trn)

print("Acuuracy",clfR.score(x_tst,y_tst))
print("confusion matrix is \n",confusion_matrix(y_tst,clfR.predict(x_tst)))

give_roc(clfR,x_tst,y_tst)

cv_score2 = cross_val_score(estimator=RandomForestClassifier(n_estimators=190),X=x_trn,y=y_trn,cv=5)
cv_score2

sns.boxplot(cv_score2)

est = {i:RandomForestClassifier(n_estimators = i) for i in range(50,300,20)}

l = []
for i,estimator in est.items():
  estimator.fit(x_trn,y_trn)
  a = estimator.predict(x_tst)
  l.append(accuracy_score(a,y_tst))

sns.lineplot(x=list(est.keys()),y=l)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')





