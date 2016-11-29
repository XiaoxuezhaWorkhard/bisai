# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

DIM_NUM = 40   #pca维数

df_train = pd.read_csv('./data/train.csv',names=['CONS_NO','LABEL'])
df_test = pd.read_csv('./data/test.csv',names=['CONS_NO'])


def chuli_id(train_str,test_str,num=10):
    train_str = np.array(train_str).astype(np.int64)
    test_str = np.array(test_str).astype(np.int64)
   
    train_str = train_str.astype('|S10')
    test_str = test_str.astype('|S10')
    test_str = test_str.reshape((len(test_str),))

    l_t = []
    nn = 1
    for str1 in train_str:
        str1 = str(str1)
        str1 = str1.rjust(num)
        str1 = str1[-num:]
        l = []
        for n in range(num):
            if str1[n] == ' ':
                l.append('10')
            elif str1[n] == '.':
                print str1
            else:
                l.append(str1[n])

        l_t.append(l)
        nn = nn+1

    l_e = []
    nn =1
    for str2 in test_str:
        str2 = str(str2)
        str2 = str2.rjust(num)
        str2 = str2[-num:]
        l = []
        for n in range(num):
            if str2[n] == ' ':
                l.append('10')
            else:
                l.append(str2[n])
        l_e.append(l)
        nn = nn+1

    t_ar = np.array(l_t, dtype=float)
    te_ar = np.array(l_e, dtype=float)

    all_arr = np.concatenate((t_ar,te_ar)) 
    enc = preprocessing.OneHotEncoder()
    enc.fit(all_arr)

    train_arr = enc.transform(t_ar).toarray()
    test_arr = enc.transform(te_ar).toarray()

    return t_ar,te_ar

train_id,test_id = chuli_id(df_train['CONS_NO'], df_test,6)
'''
pca = PCA(n_components=DIM_NUM)
pca.fit(train_id)
train_id = pca.transform(train_id)
test_id = pca.transform(test_id)
'''

output_train = np.concatenate((np.array(df_train['CONS_NO']).reshape((-1,1)), train_id,np.array(df_train['LABEL']).reshape((-1,1))), axis=1)

output_test = np.concatenate((np.array(df_test['CONS_NO']).reshape((-1,1)), test_id), axis=1)

print output_train.shape
np.savetxt('./data/pca_train_id_1113_2.csv', output_train, delimiter=',')
np.savetxt('./data/pca_test_id_1113_2.csv', output_test, delimiter=',')
