# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import chunID as useid
from sklearn.preprocessing import Imputer


useID = True #是否使用ID
DIM_NUM = 40

df_train = pd.read_csv('./data/trainfeature1108.csv', header=None)
df_test = pd.read_csv('./data/testfeature1108.csv', header=None, skiprows=3)
train_label = pd.read_csv('./data/train.csv', names=['CONS_NO','LABEL'])
test_label = pd.read_csv('./data/test.csv', names=['CONS_NO'])


train_arr = np.array(df_train, dtype=float)
test_arr = np.array(df_test, dtype=float)

train_id, test_id = useid.chuli_id(df_train.ix[:, 0], df_test.ix[:, 0], 10)

input_train = train_arr[:, 1:-1]
input_test = test_arr[:, 1:]
label = train_arr[:, -1]

#处理inf
input_train[np.isinf(input_train)] = 0
input_test[np.isinf(input_test)] = 0

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
min_max_scaler = preprocessing.MinMaxScaler()

tran_line = make_pipeline(imp, min_max_scaler)
tran_line.fit(input_train)
input_train=tran_line.transform(input_train)#变换训练集

tran_line.fit(input_test)
input_test=tran_line.transform(input_test)#变换测试集

clf = ExtraTreesClassifier(max_depth=10)
clf = clf.fit(input_train, label)
print clf.feature_importances_

model = SelectFromModel(clf,prefit=True)
input_train = model.transform(input_train)
input_test = model.transform(input_test)

if useID:
    input_train = np.concatenate((input_train,train_id), axis=1)
    input_test = np.concatenate((input_test,test_id), axis=1)

pca = PCA(n_components = DIM_NUM)
pca.fit(input_train)
input_train = pca.transform(input_train)#变换训练集
input_test = pca.transform(input_test)#变换测试集


train_result = np.concatenate((train_arr[:, 0].reshape((-1, 1)), input_train, label.reshape((-1, 1))), axis=1)
test_result = np.concatenate((test_arr[:, 0].reshape((-1, 1)), input_test), axis=1)
np.savetxt('./data/pca_train_id_1108_2.csv', train_result, delimiter=',')
np.savetxt('./data/pca_test_id_1108_2.csv', test_result, delimiter=',')


