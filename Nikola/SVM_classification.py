##
## Programmer: Nikola Andric
## Email: namdd@mst.edu
## Last Eddited: 11/07/2021
##
##

from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

my_dataset = pd.read_csv(
    r'./waters_datasets/Cleaned_Datasets/merged_classification_dataset_training.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"]]

my_dataset_test = pd.read_csv(
    r'./waters_datasets/Cleaned_Datasets/testing_data_1.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"]]

x_train = my_dataset[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI clf"]]
y_train = my_dataset["WQI clf"]

x_test = my_dataset_test[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI clf"]]
y_test = my_dataset_test["WQI clf"]


clf = svm.SVC()
clf.fit(x_train, y_train)


prediction = clf.predict(x_test)



# get the accuracy
print("The accuracy score before hypertuning is: ",accuracy_score(y_test, prediction))


#hypertuning
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.0001,0.001, 0.01, 0.1, 1, 10, 10]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs = -1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_, grid_search.best_score_

hypers, score = svc_param_selection(x_train, y_train, 5)


print(hypers, score)


clf = svm.SVC(C=hypers['C'],gamma=hypers['gamma'],kernel='linear')
clf.fit(x_train,y_train)
pred = clf.predict(x_test)

# get the accuracy
print("The accuracy score after hypertuning is: ",accuracy_score(y_test, pred))

print (confusion_matrix(y_test, pred))

pred = pred.astype(int)

# Save the predictinos into a csv file so you can add them to the orginal file.
np.savetxt("predictions_SVM_test_1.csv", pred)
