import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.cross_validation import train_test_split 
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from cmath import sqrt

data = {}

def formatOutput (InputFileName, predLabel, encoderFile):
    testId = pd.read_csv(InputFileName,
            usecols = ['id'])
    testId_rep = np.repeat(testId['id'].values, 5)
    encoder = pd.read_csv(encoderFile, header = None)
    
    label = pd.DataFrame(predLabel) 
    testcountry = label.apply(lambda x: encoder[0][x])

    result = pd.DataFrame({'id': testId_rep, 
        'country': testcountry.values[:, 0]})
    result.to_csv('submission.csv', header=True, index=False, columns=['id', 'country'])
    return

def cleanData (data):
    del data['id']
    del data['date_first_booking']
    ################### convert date to numeric ##############
    featureNames = data.columns.values

    name = 'date_account_created'
    addColNames = [name+'_mm', name+'_dd', name+'_yy']
    foo = lambda x: pd.Series([i for i in x.split('/')])
    data[addColNames] = data[name].apply(foo)
    del data[name]

    name = 'timestamp_first_active'
    addColNames = [name+'_yy', name+'_mm', name+'_dd']
    foo = lambda x: pd.Series([i for i in [x[2:4],x[4:6], x[6:8]]]) 
    data[addColNames] = data[name].astype(int).astype(str).apply(foo)
    del data[name]

    featureNames = np.delete(featureNames, [0,1])
    ############## imputation #################  
    tmp = data['age'].dropna()
    meanVal = tmp.mean()
    data['age'].fillna(meanVal, inplace = True)
    
    data.fillna('-unknown-', inplace = True)
    ############ encoding categorical features ####################
    for col in featureNames:
        if data[col].dtype != np.float64 and data[col].dtype != np.int64: 
            print(col)
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col])
            print(le.classes_)
            if col == 'country_destination':
                np.savetxt("country_destination_encoder_list.csv", 
                        le.classes_, delimiter=",", fmt = '%s')

def dataPrep ():
    trainData = pd.read_csv('train_users.csv', header = 0)
    cleanData(trainData)
    trainData.to_csv('cleaned_train_data.csv', index = False)

    testData = pd.read_csv('test_users.csv', header = 0)
    cleanData(testData)
    testData.to_csv('cleaned_test_data.csv', index = False)

def knnClassify (X, Y):
    clf = neighbors.KNeighborsClassifier(n_neighbors = 15)
    clf.fit(X, Y)
    return clf

def logisticRegressionClassify (X, Y, c_i):
    clf = LogisticRegression(C = c_i, fit_intercept = True) 
    clf.fit(X, Y)
    return clf

def plotLearningCurve(x, y1, y2):
    line_up, = plt.plot(x, y1, 'r', label = 'Train')
    line_down, = plt.plot(x, y2, 'b', label = 'Test')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.legend(handles=[line_up, line_down])
    plt.savefig('lc.pdf')

def main():
#    dataPrep()
    trainData = pd.read_csv('cleaned_train_data.csv')
    testData = pd.read_csv('cleaned_test_data.csv')
    ########## remove features with zero variance #################
    featureNames = testData.columns.values
    for name in featureNames:
        if testData[name].var() == 0:
            print(name)
            del trainData[name]
            del testData[name]
    ############### convert dataframe to ndarray ##################
    Y = trainData['country_destination'].values
    del trainData['country_destination']
    trainData = trainData.values
    testData = testData.values
    ############# Standardization ################################
    X = preprocessing.scale(trainData)
    testData = preprocessing.scale(testData)
    ###################### split training set #####################
#    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
#            test_size = 0.3, random_state = 1)
    ######################### classification #####################
    clf = logisticRegressionClassify(X, Y, 1)
    predictLabel = np.empty([0,0], dtype = int)
    label_prob = clf.predict_proba(testData)
    for row_i in range(0, testData.shape[0]):
        predictLabel = np.append(predictLabel, 
                label_prob[row_i, :].argsort()[::-1][:5])
    predictLabel.astype(int)
    ################### Output result ##########################
    formatOutput('test_users.csv', predictLabel, 
            'country_destination_encoder_list.csv')

    data['X'] = X
    data['Y'] = Y
    data['test'] = testData
    data['clf'] = clf
    data['label'] = predictLabel

if __name__=='__main__':
    main()
