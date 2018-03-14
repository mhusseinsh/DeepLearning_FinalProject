import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob
import json
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(source_dir='./final_project'):
    
    configs = []
    learning_curves = []
    
    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])
            learning_curves.append(tmp['learning_curve'])
    return(configs, learning_curves)
configs, learning_curves = load_data()

for row in learning_curves:
	del row[0:-1]
X = np.zeros((len(configs), len(configs[0])))
Y = np.asarray(learning_curves)

for row, config in enumerate(configs):
	X[row, 0] = config['batch_size']
	X[row, 1] = config['log2_n_units_2']
	X[row, 2] = config['log10_learning_rate']
	X[row, 3] = config['log2_n_units_3']
	X[row, 4] = config['log2_n_units_1']

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)


kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))