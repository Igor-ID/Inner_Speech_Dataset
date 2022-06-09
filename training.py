import numpy as np
import pandas as pd
from Python_Processing.Data_extractions import Extract_data_from_subject
from Python_Processing.Data_processing import Select_time_window, Transform_for_classificator
# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')


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


# Let's take subject 1 for training, subject 2 for validation and subject 3 for testing
X_train, Y_train = get_data()
X_validate, Y_validate = get_data(subject=2)
X_test, Y_test = get_data(subject=3)

kernels, chans, samples = 1, 128, 640

# convert labels to one-hot encodings.
Y_train = np_utils.to_categorical(Y_train)
Y_validate = np_utils.to_categorical(Y_validate)
Y_test = np_utils.to_categorical(Y_test)

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 128 channels and 640 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples,
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16,
               dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

# count number of parameters in the model
numParams = model.count_params()
print(numParams)

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during
# optimization to balance it out. This data is approximately balanced so we
# don't need to do this, but is shown here for illustration/completeness.
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0:1, 1:1, 2:1, 3:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300,
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight = class_weights)

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

probs = model.predict(X_test)
preds = probs.argmax(axis = -1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print(f"probs {probs}")
print(f"predictions {preds}")
print("Classification accuracy: %f " % acc)