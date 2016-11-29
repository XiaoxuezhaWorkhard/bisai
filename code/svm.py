import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV

traindata = './data/pca_train_id_1102_1.csv'
tstdata = './data/pca_test_id_1102_1.csv'
train = np.loadtxt(traindata,delimiter=',')
test = np.loadtxt(tstdata,delimiter=',')

label = train[:,-1]

ratio = float(np.sum(label==0)) / np.sum(label==1)

clf = svm.SVC(C=1, probability = True)

param = {'kernel': ['rbf'], 'gamma': [0.1, 0.2],'C': [1, 10,100,1000]}

gsearch = GridSearchCV(clf, param, cv=3, scoring='average_precision', verbose=True,n_jobs=-1 )
gsearch.fit(train[:,1:-1], label)

print gsearch.best_score_, gsearch.best_params_

ypred = gsearch.predict_proba(test[:,1:])[:,1]

result = np.concatenate((test[:,0].reshape((-1,1)), ypred.reshape((-1,1))), axis=1)

result_sort = sorted(result, key = lambda result:result[1], reverse=True)

np.savetx('./result/1102_2.csv', np.array(result_sort[:,0]), delimiter=',')


 
