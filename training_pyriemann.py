import numpy as np
import pandas as pd
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Data_processing import Select_time_window, Transform_for_classificator
# mne imports
import mne
from mne import io
from mne.datasets import sample

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
# tools for plotting confusion matrices
from matplotlib import pyplot as plt


def get_data(root_dir="data", datatype="EEG", subject=1, t_start=1, t_end=3.5, fs=256):
    # Load all trials for a single subject
    X, Y = Extract_data_from_subject(root_dir, subject, datatype)

    # Cut useful time. i.e action interval
    X = Select_time_window(X=X, t_start=t_start, t_end=t_end, fs=fs)

    # Conditions to compared
    conditions = [["Inner"], ["Inner"], ["Inner"], ["Inner"]]
    # The class for the above condition
    classes = [["Up"], ["Down"], ['Left'], ['Right']]

    # Transform data and keep only the trials of interest
    X, Y = Transform_for_classificator(X, Y, classes, conditions)

    return X, Y


# Let's take subject 1 for training, subject 2 for testing
X_train, Y_train = get_data()
X_test, Y_test = get_data(subject=2)

chans, samples = 128, 640

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 128 channels and 640 time-points.
X_train = X_train.reshape(X_train.shape[0], chans, samples)
X_test = X_test.reshape(X_test.shape[0], chans, samples)

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

# preds_rg = np.zeros(len(Y_test))

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, Y_train)
preds_rg = clf.predict(X_test)

# Printing the results
acc2 = np.mean(preds_rg == Y_test)
print("Classification accuracy: %f " % acc2)

names = ['Up', 'Down', 'Left', 'Right']

plt.figure()
plot_confusion_matrix(preds_rg, Y_test, names, title = 'xDAWN + RG')

plt.show()