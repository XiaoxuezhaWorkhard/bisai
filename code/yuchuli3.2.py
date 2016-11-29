# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.externals.six import StringIO
from sklearn import tree
from IPython.display import Image
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
df_train = pd.read_csv('./data/trainfeature_shao.csv',header=None)
df_test = pd.read_csv('./data/testfeature_shao.csv',header=None)
train_diff = pd.read_csv('./data/trainfeature_diff1_shao.csv',header=None)
test_diff = pd.read_csv('./data/testfeature_diff1_shao.csv',header=None)
train_diff2 = pd.read_csv('./data/trainfeature_diff2_shao.csv',header=None)
test_diff2 = pd.read_csv('./data/testfeature_diff2_shao.csv',header=None)
train_label = pd.read_csv('./data/train.csv',header=None)
#train_id = pd.read_csv('/usr/local/hadoop/src/data/yichangzhi/trainid.csv',header=None)
#test_id = pd.read_csv('/usr/local/hadoop/src/data/yichangzhi/testid.csv',header=None)

#train_id_arr = np.array(train_id)
#test_id_arr = np.array(test_id)


#feature_names = df.columns[1:]
#target_names = np.array(['kaifa','shipingyingyue','tongxin','bangong'],dtype='|S10')
def gouzao_id(train_str,test_str,num=3):
    train_str = np.array(train_str,dtype='|S10')
    test_str = np.array(test_str,dtype='|S10')


    l_t = []
    for str1 in train_str:
        st = str1[-num:]
        l_t.append(st)
        print st

    l_e = []
    for str2 in test_str:
        st = str2[-num:]
        l_e.append(st)
        print st

    t_ar = np.array(l_t)
    te_ar = np.array(l_e)
    t_ar = t_ar.reshape((len(t_ar),1))
    te_ar = te_ar.reshape((len(te_ar),1))

    all_arr = np.concatenate((t_ar,te_ar))

    enc = preprocessing.OneHotEncoder()
    enc.fit(all_arr)

    train_arr = enc.transform(t_ar).toarray()
    test_arr = enc.transform(te_ar).toarray()

    return train_arr,test_arr

def paixu(arr_in,num=0):
    a_arg = np.argsort(arr_in[:,num])
    arr_in = arr_in[a_arg]
    return arr_in




train_arr = np.array(df_train,dtype=float)
test_arr = np.array(df_test,dtype=float)

train_diff_arr = np.array(train_diff,dtype=float)
test_diff_arr = np.array(test_diff,dtype=float)

train_diff2_arr = np.array(train_diff2,dtype=float)
test_diff2_arr = np.array(test_diff2,dtype=float)

train_diff_arr[np.isinf(train_diff_arr)] = 0
test_diff_arr[np.isinf(test_diff_arr)] = 0
train_diff2_arr[np.isinf(train_diff2_arr)] = 0
test_diff2_arr[np.isinf(test_diff2_arr)] = 0

train_label = np.array(train_label,dtype=float)
#paixu
#train_arr = paixu(train_arr)
#test_arr = paixu(test_arr)
#train_diff_arr = paixu(train_diff_arr)
#test_diff_arr = paixu(test_diff_arr)
#train_label = paixu(train_label)


input_train = np.array(train_arr[:,1:],dtype=float)
label_train = train_arr[:,0] #id
label2_train = train_label[:,1]
input_test = np.array(test_arr[:,1:],dtype=float)
label_test = test_arr[:,0] #id



train_id,test_id = gouzao_id(label_train,label_test,3)

input_train = np.concatenate((input_train,train_diff_arr[:,1:],train_diff2_arr[:,1:]),axis=1)
input_test = np.concatenate((input_test,test_diff_arr[:,1:],test_diff2_arr[:,1:]),axis=1)
#删除方差较小的维度
#sel = VarianceThreshold(threshold=0.05)
#input_train = sel.fit_transform(input_train)
#print sel.variances_
#print '删除较小方差维度：'
#print input_train.shape
#用决策树进行特征选择
'''
print '使用决策树进行属性选择：'
print input_train.shape
clf = ExtraTreesClassifier(max_depth=10)
clf = clf.fit(input_train,label_train)
print clf.feature_importances_
#tree = clf.tree_
#Image(graph1.create_png())
model = SelectFromModel(clf,prefit=True)
input_train = model.transform(input_train)
input_test = model.transform(input_test)
'''
print input_train.shape
#def cov_type():
#input_test = np.array(df2.ix[:,0:-1]).copy()
#label_test = np.array(df2.ix[:,-1]).copy()
DIM_NUM = 45
pca = PCA(n_components=DIM_NUM)
#input_train = pca.fit(input_train).transform(input_train)
min_max_scaler = preprocessing.MinMaxScaler()
#
#print x_r.shape
#lda = LinearDiscriminantAnalysis(n_components=30)
tran_line = make_pipeline(min_max_scaler)
tran_line.fit(input_train)
input_train=tran_line.transform(input_train)#变换训练集
input_test=tran_line.transform(input_test)#变换测试集



#单变量属性选择
clf = SelectKBest(chi2,k=50)

clf.fit(input_train,label_train)
input_train = clf.transform(input_train)
input_test = clf.transform(input_test)

'''
print '使用决策树进行属性选择：'
print input_train.shape
#clf = DecisionTreeClassifier(max_depth=10)
clf = ExtraTreesClassifier(max_depth=10)
clf = clf.fit(input_train,label_train)
print clf.feature_importances_
#tree = clf.tree_
#Image(graph1.create_png())
model = SelectFromModel(clf,threshold="mean",prefit=True)
input_train = model.transform(input_train)
input_test = model.transform(input_test)
'''
input_train = np.concatenate((input_train,train_id),axis=1)
input_test = np.concatenate((input_test,test_id),axis=1)


print input_train.shape
tran_line = make_pipeline(pca)
tran_line.fit(input_train)
input_train=tran_line.transform(input_train)#变换训练集
input_test=tran_line.transform(input_test)#变换测试集


print(pca.explained_variance_ratio_)

#lda_scaler = lda.fit(input_train,label_train)
#input_train = lda_scaler.transform(input_train)
#print input_train.shape
#test_x = lda_scaler.transform(input_test)
arr_train = np.empty((len(input_train),DIM_NUM+2))
arr_test = np.empty((len(input_test),DIM_NUM+1))
#arr2 = np.empty((len(input_test),6+1))
arr_train[:,1:-1] = input_train
arr_train[:,0] = label_train
arr_train[:,-1] = label2_train
arr_test[:,1:] = input_test
arr_test[:,0] = label_test

np.savetxt('./data/pca_train_id_1030_1.csv',arr_train,delimiter=',')
np.savetxt('./data/pca_test_id_1030_1.csv',arr_test,delimiter=',')
