import os
import sys
import glob
import json

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils

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

def prepare_data(configs, learning_curves):

	for row in learning_curves:
		del row[0:-1]


	Y = np.asarray(learning_curves)

	X = np.zeros((265, 5))

	for row, config in enumerate(configs):
		X[row,0] = config['batch_size'] 
		X[row,1] = config['log2_n_units_2'] 
		X[row,2] = config['log10_learning_rate'] 
		X[row,3] = config['log2_n_units_3'] 
		X[row,4] = config['log2_n_units_1'] 
	return X, Y

X, y = prepare_data(configs, learning_curves)

def baseline():
	print("In the baseline...")
	model = Sequential()
	model.add(Dense(5, input_dim = 5, kernel_initializer = 'normal', activation = 'relu'))
	#model.add(Dense(20, kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model

def evaluate_raw_data(X, y, baseline):
	seed = 7
	np.random.seed(seed)
	estimator = KerasRegressor(build_fn = baseline, nb_epoch = 100, batch_size  = 5)

	kfold = KFold(n_splits = 3, random_state = seed)
	results = cross_val_score(estimator, X, y, cv = kfold)
	print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def evaluate_preprocessed_data_pipeline(X, y, baseline):
	seed = 7
	np.random.seed(seed)
	estimator = []
	estimator.append(('preprocess', StandardScaler()))
	estimator.append(('mlp', KerasRegressor(build_fn = baseline, nb_epoch = 100, batch_size  = 5)))
	pipeline = Pipeline(estimator)
	
	kfold = KFold(n_splits = 3, random_state = seed)
	results = cross_val_score(pipeline, X, y, cv = kfold)
	print("Results (pipeline with preprocessed data): %.2f (%.2f) MSE" % (results.mean(), results.std()))

def evaluate_preprocessed_data(X, y, baseline):
	seed = 7
	np.random.seed(seed)

	standarize = StandardScaler()
	X_new = standarize.fit_transform(X ,y)
	estimator = KerasRegressor(build_fn = baseline, nb_epoch = 100, batch_size  = 5)
	kfold = KFold(n_splits = 3, random_state = seed)
	results = cross_val_score(estimator, X_new, y, cv = kfold)
	print("Results (without pipeline): %.2f (%.2f) MSE" % (results.mean(), results.std()))

evaluate_raw_data(X, y, baseline)
evaluate_preprocessed_data_pipeline(X, y, baseline)
evaluate_preprocessed_data(X, y, baseline)













