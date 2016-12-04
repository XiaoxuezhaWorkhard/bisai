# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def modelfit(train_arr,label_arr,test_arr,useTrainCV=True,cv_folds=3,early_stopping_rounds=10):

    if useTrainCV:
        dtrain = xgb.DMatrix(train_arr,label_arr)
        label = dtrain.get_label()
   #     ratio = float(np.sum(label==0)) / np.sum(label==1)
    '''
        alg.set_params(scale_pos_weight=ratio)

        xgb_param = alg.get_xgb_params()
	skf = StratifiedKFold(n_splits=2,shuffle=True)
        cvresult = xgb.cv(xgb_param,dtrain,num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,stratified=True,
                          metrics='map',early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        print cvresult.shape[0]
        alg.set_params(n_estimators=cvresult.shape[0])
    '''
    param_test1 = {
        'eta':0.01,
        'max_depth':6,
        'early_stopping_rounds':20,
        'min_child_weight':3,
        'colsample_bytree':0.8,
        'objective':'binary:logistic',
        'max_delta_step':2,
        'scale_pos_weight':ratio,
        'eval_metric': 'map',
        'seed':27
    }
    dtest = xgb.DMatrix(test_arr)
    clf = svm.SVC(C=1)
    train, val, train_y, val_y = train_test_split(train_arr,label_arr, test_size = 0.3,random_state=27)

    dval = xgb.DMatrix(val, val_y)
    dtrain2 = xgb.DMatrix(train, train_y)

    watchlist = [(dval,'val'),(dtrain2,'train')]
    model = xgb.train(param_test1, dtrain, num_boost_round=1600, evals=watchlist)

#    gsearch1 = GridSearchCV(estimator=alg1,param_grid = param_test1,scoring='average_precision',iid=False,n_jobs=-1,cv=3)
    dtrain_predictions = model.predict(dtest,ntree_limit=model.best_ntree_limit)
#    model.save_model('./model/xgb.model')


  #  gsearch1.fit(train_arr,label_arr)

  #  print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_




#    alg.fit(train_arr,label_arr,eval_metric='map')
#    dtrain_predprob = alg.predict_proba(test_arr)[:,1]
    return dtrain_predictions

def xgboost_f(traindata, tstdata, traindata1, tstdata1):
    arr1 = np.loadtxt(traindata,delimiter=',',skiprows=0)  #训练数据
    arr2 = np.loadtxt(tstdata,delimiter=',')
    arr1_1 = np.loadtxt(traindata1,delimiter=',',skiprows=0)  #训练数据
    arr2_1 = np.loadtxt(tstdata1,delimiter=',')

    label = np.array(arr1[:,-1],dtype=int)

#    dtrain_predictions = modelfit(arr1[:,1:-1],label_arr,arr2[:,1:])
    ratio = float(np.sum(label==0)) / np.sum(label==1)


    param_test1 = {
        'eta':0.02,
        'max_depth':6,
        'colsample_bytree':0.7,
        'objective':'binary:logistic',
        'eval_metric':'map',
        'scale_pos_weight':ratio,
        'seed':27
    }
    dtrain = xgb.DMatrix(arr1[:,1:-1],arr1[:,-1])
    dtest = xgb.DMatrix(arr2[:,1:])

    dtrain1 = xgb.DMatrix(arr1_1[:,1:-1],arr1_1[:,-1])
    dtest1 = xgb.DMatrix(arr2_1[:,1:])

    train, val, train_y, val_y = train_test_split(arr1[:,1:-1],label, test_size = 0.3,random_state=27)

    dval = xgb.DMatrix(val, val_y)
    dtrain2 = xgb.DMatrix(train, train_y)

    watchlist = [(dval,'val'),(dtrain2,'train')]


    model = xgb.train(param_test1, dtrain, num_boost_round=1000, evals=watchlist)

    pred1 = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    model1 = xgb.train(param_test1, dtrain1, num_boost_round=1000)

    pred2 = model1.predict(dtest1, ntree_limit=model1.best_ntree_limit)

    pred = 0.75*pred1 + 0.2*pred2


    pred = np.concatenate((arr2[:, 0].reshape((-1, 1)), pred.reshape((-1, 1))), axis=1)

#    print dtrain_predictions

#    for n in [493508051, 1332987957, 5340009393, 6262373557]:
#        pred[pred[:,0]==n][:,1] = 0.85

#    a_arg = np.argsort(-pred[:,1])

#    pred = pred[a_arg]




#    new_arr = new_arr[::-1]

    np.savetxt('result1113_2.csv',pred,delimiter=',')



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

if __name__ == '__main__':
    traindata='./data/pca_train_id_1113_2.csv'
    tstdata='./data/pca_test_id_1113_2.csv'
    traindata1 ='./data/pca_train_id_1113_1.csv'
    tstdata1 ='./data/pca_test_id_1113_1.csv'
    xgboost_f(traindata, tstdata, traindata1, tstdata1)






