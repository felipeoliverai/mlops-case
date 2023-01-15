import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings('ignore')


def soloOrNot(SibSp,Parch):
    if(SibSp+Parch>0):
        return True
    else:
        return False


def train(): 

    # load dataset
    df = pd.read_csv("titanic.csv")

    # preprocessing 
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].value_counts().idxmax())
    df=df.drop(['Cabin','Ticket','Name'],axis=1)
    df.isnull().sum()
    df['Solo'] = df[['SibSp','Parch']].apply(lambda df: soloOrNot(df['SibSp'],df['Parch']),axis=1)
    df=df.drop(['SibSp','Parch'],axis=1)
    df=df.drop(["PassengerId"], axis=1)
    df=pd.get_dummies(df, columns=["Pclass","Embarked","Sex"],drop_first=True)
    X = df.drop('Survived',axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)


    # train model 
    log_model = LogisticRegression()
    log_model.fit(X_train,y_train)

    # save model 
    filename = 'model/model_titanic.pkl'
    pickle.dump(log_model, open(filename, 'wb'))


if __name__ == "__main__": 
    train()