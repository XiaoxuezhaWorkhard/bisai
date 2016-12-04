import pandas as pd
from pandas.tools.plotting import radviz
import matplotlib.pyplot as plt

data = pd.read_csv('../data/train1203_1.csv')

radviz(data.drop(data.columns[0], axis=1), data.columns[-1])
plt.show()
plt.savefig('../data/rad.png')

