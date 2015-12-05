import csv as csv
import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.cross_validation import KFold
from sklearn import neighbors
from cmath import sqrt

data = {}

def loadInput (fileName):
    data = pd.read_csv(fileName, header = 0)
    return data

def cleanData (data):
    del data['id']
    ################### convert date to numeric ##############
    featureNames = data.columns.values
    dateInd = list()
    ind = 0
    for name in featureNames:
        if name.find('date') != -1:
            if data[name].isnull().values.any() == 1:
                data[name].fillna('0/0/0', inplace = True)
            addColNames = [name+'_mm', name+'_dd', name+'_yy']
            foo = lambda x: pd.Series([i for i in x.split('/')])
            data[addColNames] = data[name].apply(foo)
            dateInd.append(ind)
            del data[name]
        ind += 1
#    print(dateInd)
    featureNames = np.delete(featureNames, dateInd)
    ############## imputation #################  
    tmp = data['age'].dropna()
    meanVal = tmp.mean()
    data['age'].fillna(meanVal, inplace = True)
    
    data.fillna('empty', inplace = True)
    ############ encoding categorical features ####################
    le = preprocessing.LabelEncoder()
    for col in featureNames:
        if data[col].dtype != np.float64 and data[col].dtype != np.int64: 
            print(col)
            data[col] = le.fit_transform(data[col])
    return data  

def writeOutput (fileName, data):
    data.to_csv(fileName, sep = ',') 
    return

def knnclassify (X, Y):
    accuracy = []
    cv = KFold(n = len(X), n_folds = 5)
    for train, test in cv:
        x_train = X[train, :]
        y_train = Y[train]
        x_test = X[test, :]
        y_test = Y[test]
        clf = neighbors.KNeighborsClassifier(n_neighbors = 13)
        clf.fit(x_train, y_train)
        accuracy.append(clf.score(x_test, y_test))
    print("Mean accuracy = %.5f\t std dev = %.5f" % (np.mean(accuracy), np.std(accuracy)))
    clf.fit(X, Y)
    return clf

def main():
#    trainData = loadInput('test_users.csv')
#    trainData = cleanData(trainData)
#    writeOutput('cleaned_test_data.csv', trainData)
    trainData = loadInput('cleaned_train_data.csv')
    testData = loadInput('cleaned_test_data.csv')

    ########## remove features with zero variance #################
    del trainData['date_account_created_yy']
    del testData['date_account_created_yy']
    ############### convert dataframe to ndarray ##################
    del trainData['Unnamed: 0']
    del testData['Unnamed: 0']

    trainY = trainData['country_destination'].values
    del trainData['country_destination']
    trainData = trainData.values
    testData = testData.values
    ############# Standardization ################################
    trainData = preprocessing.scale(trainData)
    testData = preprocessing.scale(testData)
    ######################### knn classification #################
    clf = knnclassify(trainData, trainY)
    predictLabel = clf.predict(testData)
    result = pd.DataFrame({'label':predictLabel})
    writeOutput('test_data_label.csv', result)

    data['train'] = trainData
    data['trainY'] = trainY
    data['test'] = testData

if __name__=='__main__':
    main()
