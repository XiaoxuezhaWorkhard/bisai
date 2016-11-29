# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('./data/df_train.csv',parse_dates=['DATA_DATE'])
df_test = pd.read_csv('./data/df_test.csv',parse_dates=['DATA_DATE'])

#看一下变量持续的时间
dic =  {'train_data':df_train,'test_data':df_test}
for d in dic:
    print('Strat of '+ d +':' + str(dic[d]['DATA_DATE'].min()))
    print('End of '+ d +':' + str(dic[d]['DATA_DATE'].max()))
    print('Range of '+ d +':' + str(dic[d]['DATA_DATE'].max()-dic[d]['DATA_DATE'].min())+'\n')

fig = plt.figure()
'''
ax1 = fig.add_subplot(1, 1, 1)

data_1 = pd.DataFrame()
#data_x['dianliang'] = df_train.groupby('DATA_DATE')['KWH'].sum()
data_1['yichangmean'] = df_train.groupby('DATA_DATE')['LABEL'].mean()
data_1['toudiansum'] = df_train.groupby('DATA_DATE')['LABEL'].sum()

data_1.plot(ax=ax1,secondary_y='yichangmean')
'''

def fun(arr):
    ts = pd.Series(np.array(arr['KWH']),index=arr['DATA_DATE'])
    ts = ts.diff(1)
    ts_min = ts.min()
    ts_max = ts.max()
    ts_jizhi = ts_max-ts_min
    ts_std = ts.std()
    label = int(pd.unique(arr['LABEL']))

    return pd.DataFrame([ts_jizhi,ts_std,label])


data_2 = df_train.groupby('CONS_NO').apply(fun)

data_2 = data_2.unstack()
data_2 = pd.DataFrame(np.array(data_2),index = data_2.index,columns=['JIZHI','STD','LABEL'])

ax2 = fig.add_subplot(1,1,1)

ax3=sns.boxplot(x="LABEL",y="STD",data=data_2)

ax3.set_yticks(range(0,100,10))
ax3.set_ylim([0,100])
sns.FacetGrid(data_2, hue="LABEL", size=6).map(sns.kdeplot, "JIZHI").add_legend()
sns.FacetGrid(data_2, hue="LABEL", size=6).map(sns.kdeplot, "STD").add_legend()
sns.FacetGrid(data_2, hue="LABEL", size=10).map(plt.scatter, "JIZHI", "STD").add_legend()

plt.show()
