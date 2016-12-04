# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,ShuffleSplit

def modelfit(param,dtrain,dtest,arr_train,useTrainCV=True,cv_folds=3,early_stopping_rounds=50):
    if useTrainCV:
	skf = StratifiedKFold(n_splits=2,shuffle=True)
        cvresult = xgb.cv(param,dtrain,num_boost_round=2000,nfold=cv_folds,stratified=True,
                          metrics='map',early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        print cvresult.shape[0]

#    alg1 = XGBClassifier()
 #   gsearch1 = GridSearchCV(estimator=alg1,param_grid = param_test1,scoring='average_precision',iid=False,n_jobs=-1,cv=3)

  #  gsearch1.fit(train_arr,label_arr)

  #  print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


    train_label = dtrain.get_label()

    ratio  = float(np.sum(train_label==1)) / np.sum(train_label==0)

    skf = StratifiedShuffleSplit(n_splits=1, random_state=0, test_size = 1-ratio)



    for train_index, test_index in skf.split(arr_train[train_label==0], train_label[train_label==0]):

        dtrain_x, dtrain_y = arr_train[train_label==0][train_index], train_label[train_label==0][train_index]

    train_x = np.concatenate((arr_train[train_label==1], dtrain_x))
    train_y = np.concatenate((train_label[train_label==1], dtrain_y))
    dtrain2 = xgb.DMatrix(train_x, train_y)
    bst = xgb.train(param,dtrain,cvresult.shape[0])

    bst.save_model("bst.model")

    dtrain_predictions = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
#    dtrain_predprob = bst.predict_proba(dtest)[:,1]
    return dtrain_predictions

def xgboost_f(traindata,tstdata):
    arr1 = np.loadtxt(traindata,delimiter=',',skiprows=0)  #训练数据
    arr2 = np.loadtxt(tstdata,delimiter=',')

    label_arr = np.array(arr1[:,-1],dtype=int)

    #训练集和测试集
    dtrain = xgb.DMatrix(arr1[:,1:-1],label_arr)
    dtest = xgb.DMatrix(arr2[:,1:])

    param_test1 = {
        'booster':'gbtree',
        'gamma':1,
        'eta':0.02,
        'max_depth':6,
        'min_child_weight':10,
        'subsample':0.7,
        'colsample_bytree':0.8,
        'reg_alpha':0.05,
        'objective':'binary:logistic',
        'seed':27

    }

    def fpreproc(dtrain, dtest, param):
        label = dtrain.get_label()
        ratio = float(np.sum(label == 0)) / np.sum(label==1)
        param['scale_pos_weight'] = ratio
        return (dtrain, dtest, param)

    dtrain,dtest,param = fpreproc(dtrain,dtest,param_test1)


    dtrain_predictions = modelfit(param,dtrain,dtest,arr1[:,1:-1])


#    dtrain = xgb.DMatrix(arr1[:,1:-1],label=label_arr)

 #   dtest = xgb.DMatrix(arr2[:,1:])
 #   param = {'max_depth':5, 'eta':0.3, 'silent':0, 'objective':'binary:logistic',
 #            'nthread':8}
 #   num_round=30

  #  print ('running cross validation')

 #   res = xgb.cv(param,dtrain,num_boost_round=50,nfold=20,
  #               metrics={'map'},seed=40,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

   # print (res)

   # watchlist = [(dtrain,'train')]
  #  bst = xgb.train(param,dtrain,num_round,evals=[(dtrain,'train'),('eval_metric,','map')],verbose_eval=True)



   # bst.save_model('0001.model')

  #  ypred=bst.predict(dtest)

    new_arr = np.empty((arr2.shape[0],2))

    print dtrain_predictions

    new_arr[:,0] = arr2[:,0]
    new_arr[:,1] = dtrain_predictions

    a_arg = np.argsort(new_arr[:,1])

    new_arr = new_arr[a_arg]

    new_arr = new_arr[::-1]

    np.savetxt('result1105_1.csv',new_arr,delimiter=',')

if __name__ == '__main__':
    traindata='./data/pca_train_id_1105_1.csv'
    tstdata='./data/pca_test_id_1105_1.csv'
    xgboost_f(traindata,tstdata)






