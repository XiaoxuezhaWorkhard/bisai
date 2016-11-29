# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


reader = pd.read_csv('../data/ALL_USER_YONGDIAN_DATA.zip', compression='zip', iterator=True)
loop = True
chunkSize = 10000000
chunks = []

while loop:
    try:
        chunk = reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print "Iteration is stopped."
data = pd.concat(chunks, ignore_index=True)
print data.columns
print 1
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
print 2
df_train = pd.merge(data, train)
df_train.to_csv('../data/df_train.csv', index = False)
del df_train
print 3
df_test = pd.merge(data, test)
df_test.to_csv('../data/df_test.csv', index = False)
