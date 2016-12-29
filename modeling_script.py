import pandas as pd
import csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

def load_data(filepath):
    data = pd.read_csv(str(filepath))
    return data

def transforma_data(data):
    data.drop('index', axis=1)
    data['is_female'] = data['Sex'].apply(lambda x: 1 if x=='female' else 0)
    data['embarked_s'] = data['Embarked'].apply(lambda x: 1 if x =='S' else 0)
    data['embarked_c'] = data['Embarked'].apply(lambda x: 1 if x =='C' else 0)
    data['is_child'] = data['Age'].apply(lambda x: 1 if x<=13 else 0)
    data['is_elderly'] = data['Age'].apply(lambda x: 1 if x>=60 else 0)
    return data

def define_x(data):
    x = data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Embarked',
                'Cabin', 'Sex', 'Age'], axis=1)
    return x

def define_y(data):
    y = data['Survived']
    return y

def train_model(x_train, y_train, x_test, y_test):
    et = ExtraTreesClassifier(bootstrap=False, class_weight=None,
        criterion='gini',
        max_depth=7, max_features=0.25, max_leaf_nodes=None,
        min_samples_leaf=2, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
        oob_score=False, random_state=None, verbose=0, warm_start=False)
    et.fit(x_train,y_train)
    return et.predict(x_test)

def 
