from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

my_dataset = pd.read_csv(
    r'./waters_datasets/Merged_Dataset_For_Trainging.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"]]



x = my_dataset[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI clf"]]
y = my_dataset["WQI clf"]

# normalize
scaler = StandardScaler()

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

print(type(x_train))
clf = svm.SVC()
clf.fit(x_train, y_train)


prediction = clf.predict(x_test)

# get the accuracy
print(accuracy_score(y_test, prediction))


def svc_param_selection(X, y, nfolds):
    Cs = [0.1, 1, 10,100]
    gammas = [0.0001,0.001, 0.01, 0.1, 1, 10, 10]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs = -1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_, grid_search.best_score_

hypers, score = svc_param_selection(x_train, y_train, 5)

print(hypers, score)

