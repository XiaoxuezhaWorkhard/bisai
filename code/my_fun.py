# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

def modelfit(test_arr,dim,limit):

    arr = test_arr[:,dim]



    

    

    return y_pred

def xgboost_f(traindata,tstdata):
    arr1 = np.loadtxt(traindata,delimiter=',',skiprows=0)  #训练数据
    arr2 = np.loadtxt(tstdata,delimiter=',')
    label_arr = np.array(arr1[:,-1],dtype=int)
 
    outliers_num = sum(label_arr==1)+1
    outliers_fraction = float(outliers_num)/float(len(label_arr))
#    n_samples = arr1.shape[0]

    y_pred = modelfit(xgb1,arr1[:,1:-1],label_arr,outliers_fraction,arr2[:,1:])


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

    print y_pred

    new_arr[:,0] = arr2[:,0]
    new_arr[:,1] = y_pred

    a_arg = np.argsort(new_arr[:,1])

    new_arr = new_arr[a_arg]

    new_arr = new_arr[::-1]

    np.savetxt('result1026_2.csv',new_arr,delimiter=',')

if __name__ == '__main__':
    traindata='./data/pca_train_id_1026_2.csv'
    tstdata='./data/pca_test_id_1026_2.csv'
    xgboost_f(traindata,tstdata)






