"""
==========================================
Outlier detection with several methods.
==========================================

When the amount of contamination is known, this example illustrates two
different ways of performing :ref:`outlier_detection`:

- based on a robust estimator of covariance, which is assuming that the
  data are Gaussian distributed and performs better than the One-Class SVM
  in that case.

- using the One-Class SVM and its ability to capture the shape of the
  data set, hence performing better when the data is strongly
  non-Gaussian, i.e. with two well-separated clusters;

The ground truth about inliers and outliers is given by the points colors
while the orange-filled area indicates which points are reported as inliers
by each method.

Here, we assume that we know the fraction of outliers in the datasets.
Thus rather than using the 'predict' method of the objects, we set the
threshold on the decision_function to separate out the corresponding
fraction.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats
from sklearn.externals import joblib

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
import pandas as pd
# Example settings
#outliers_fraction = 0.25
##clusters_separation = [0, 1, 2]
#df2 = pd.read_csv('/usr/local/hadoop/src/data/potplayer_pca.csv')
#pot_arr = np.array(df2.ix[:,1:]).copy()
#pot_label = np.array(df2.ix[:,0]).copy()

df = pd.read_csv('/usr/local/hadoop/src/data/QQ_pca.csv')
input_arr = np.array(df.ix[:,1:-1]).copy()
label_arr = np.array(df.ix[:,-1]).copy()
outliers_num = sum(label_arr==1)+1
outliers_fraction = float(outliers_num)/float(len(label_arr))
n_samples = input_arr.shape[0]
# define two outlier detection tools to be compared
classifiers = {
    "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction+0.2,
                                     kernel="rbf", gamma=0.1),
    "robust covariance estimator": EllipticEnvelope(store_precision=False,support_fraction=0.1,contamination=outliers_fraction)}

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(input_arr[:,0].min()-5,input_arr[:,0].max()+5, 500), np.linspace(input_arr[:,1].min()-5, input_arr[:,1].max()+5, 500))
n_inliers = int((1. - outliers_fraction)*n_samples)
n_outliers = n_samples - n_inliers
ground_truth = np.ones(n_samples, dtype=int)
ground_truth[-n_outliers:] = 0

# Fit the problem with varying cluster separation
#for i, offset in enumerate(clusters_separation):
 #   np.random.seed(42)
    # Data generation
  #  X1 = 0.3 * np.random.randn(0.5 * n_inliers, 2) - offset
   # X2 = 0.3 * np.random.randn(0.5 * n_inliers, 2) + offset
   # X = np.r_[X1, X2]
    # Add outliers
   # X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # Fit the model with the One-Class SVM
   # plt.figure(figsize=(10, 5))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit the data and tag outliers
    clf.fit(input_arr)
    if clf_name == 'robust covariance estimator':
        joblib.dump(clf,"outlier_model_QQ2.0.m")
    #y = clf.predict(input_arr).ravel()
    y_pred = clf.decision_function(input_arr).ravel()
    threshold = stats.scoreatpercentile(y_pred,
                                        100*outliers_fraction)
    y_pred = y_pred > threshold
    n_errors = (y_pred != ground_truth).sum()
    # plot the levels lines and the points
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    subplot = plt.subplot(1, 2, i + 1)
    subplot.set_title("Outlier detection")
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
    a = subplot.contour(xx, yy, Z, levels=[threshold],
                        linewidths=2, colors='red')
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                        colors='orange')
    b = subplot.scatter(input_arr[:-n_outliers, 0], input_arr[:-n_outliers, 1], c='white')
    inds1 = [i for i,yi in enumerate(pot_label) if yi==0]
    e = subplot.scatter(pot_arr[inds1,0],pot_arr[inds1,1],c='b')
    inds2 = [i for i,yi in enumerate(pot_label) if yi==1]
    f = subplot.scatter(pot_arr[inds2,0],pot_arr[inds2,1],c='red',marker='x')

   # c = subplot.scatter(input_arr[-n_outliers:, 0], input_arr[-n_outliers:, 1], c='red')
    subplot.axis('tight')
    subplot.legend(
        [a.collections[0], b,e,f],
        ['learned decision function', 'true inliers','potplayer_ceshi','wechat'],
        prop=matplotlib.font_manager.FontProperties(size=11))
    subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
    subplot.set_xlim((input_arr[:,0].min()-5, input_arr[:,0].max()+5))
    subplot.set_ylim((input_arr[:,1].min()-5, input_arr[:,1].max()+5))
plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)

plt.show()
