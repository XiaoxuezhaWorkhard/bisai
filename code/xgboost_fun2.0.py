# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

def modelfit(alg,train_arr,label_arr,test_arr,useTrainCV=True,cv_folds=3,early_stopping_rounds=50):
    if useTrainCV:
        dtrain = xgb.DMatrix(train_arr,label_arr)
        xgb_param = alg.get_xgb_params()
	skf = StratifiedKFold(n_splits=2,shuffle=True)
        cvresult = xgb.cv(xgb_param,dtrain,num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,stratified=True,
                          metrics='map',early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        print cvresult.shape[0]
        alg.set_params(n_estimators=cvresult.shape[0])

    param_test1 = {
        'gamma':[0.1,0.2],
        'learning_rate':[0.02],
        'n_estimators':[cvresult.shape[0]],
        'max_depth':[5],
        'min_child_weight':[3],
        'subsample':[0.5],
        'colsample_bytree':[0.8],
        'reg_alpha':[0.05],
        'objective':['binary:logistic'],
        'max_delta_step':[2],
        'scale_pos_weight':[0.9],
        'seed':[27]

    }
    clf = svm.SVC(C=1)

#    alg1 = XGBClassifier()
 #   gsearch1 = GridSearchCV(estimator=alg1,param_grid = param_test1,scoring='average_precision',iid=False,n_jobs=-1,cv=3)

  #  gsearch1.fit(train_arr,label_arr)

  #  print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_




    alg.fit(train_arr,label_arr,eval_metric='map')

    dtrain_predictions = alg.predict(test_arr)
#    dtrain_predprob = alg.predict_proba(test_arr)[:,1]
    return dtrain_predictions

def xgboost_f(traindata,tstdata):
    arr1 = np.loadtxt(traindata,delimiter=',',skiprows=0)  #训练数据
    arr2 = np.loadtxt(tstdata,delimiter=',')

    label_arr = np.array(arr1[:,-1],dtype=int)
    xgb1 = XGBRegressor(
         learning_rate =0.02,
         n_estimators=3000,
         max_depth=6,
         min_child_weight=1,
         gamma=0.2,
         subsample=0.6,
         colsample_bytree=0.8,
	 reg_alpha=0.05,
         objective= 'reg:linear',
         max_delta_step=1,
         scale_pos_weight=1,
         seed=27
    )
    dtrain_predictions = modelfit(xgb1,arr1[:,1:-1],label_arr,arr2[:,1:])


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

    np.savetxt('result1026_2.csv',new_arr,delimiter=',')

if __name__ == '__main__':
    traindata='./data/pca_train_id_1028_1.csv'
    tstdata='./data/pca_test_id_1028_1.csv'
    xgboost_f(traindata,tstdata)






