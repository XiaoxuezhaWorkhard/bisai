# coding: utf-8
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import pickle
train = pd.read_csv('../data/train1203_1.csv')
test =  pd.read_csv('../data/test1203_1.csv')

col_tr = train.columns
col_ts = test.columns
TRAIN = True  #是否训练
CV = True
#split train set and test set
dtrain = xgb.DMatrix(train.ix[:, col_tr[1]:col_tr[-2]], train[col_tr[-1]])
dtest = xgb.DMatrix(test.ix[:, col_ts[1]:])

clf = xgb.XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 100,
    max_depth = 8,
    colsample_bytree = 0.8,
    subsample = 0.9,
    objective = 'binary:logistic',
    min_child_weight = 1,
    gamma = 2,
    seed = 27
    )

param = clf.get_xgb_params()
if CV:
    cvresult = xgb.cv(param, dtrain, num_boost_round=2000, nfold=5, stratified=True,
                  metrics='map', early_stopping_rounds=10,verbose_eval=True)
    clf.set_params(n_estimators=cvresult.shape[0])   #set n_estimators as cv rounds
if TRAIN:
    clf.fit(train.ix[:, col_tr[1]:col_tr[-2]],train[col_tr[-1]] , eval_metric='map')
else:
    clf = pickle.load(open("ccf.pkl", "rb"))


ypred_xgb = clf.predict_proba(test.ix[:,col_ts[1]:])[:,1]
#print model report:
#print(classification_report(test_y_xgb, ypred_xgb))
#print(confusion_matrix(test_y_xgb, ypred_xgb))

#xgb.plot_importance(clf.booster())
pickle.dump(clf, open("ccf.pkl", "wb"))

result = pd.concat([test[col_ts[0]],pd.DataFrame(ypred_xgb)], axis = 1,ignore_index=True)
result.sort(result.columns[1], axis=0, ascending=False, inplace=True)

result[result.columns[0]].to_csv('result.csv', index=False,header=False)
