import os
import sys
import glob
import json
import itertools
import math
import random as ra

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, LSTM
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam

from load_prepare_data import *
from rnn import rnn_stateful 
from plotting import *


if __name__ == "__main__":

	n_estimators = [4,8,16,32]
	bootstrap = [True, False]
	min_samples_leaf = [1,2,4,8]
	max_depth = [2,4,8,16,32]
	models = 3
	epochs = [5]
	select_time = 20
	time_steps = [select_time]
	time = [5, 10, 20, 30]

	# get data
	data, targets, targets_original = prepare_data()
	data_scaled = preprocess_data(data)
	targets_selected = y_select(targets_original, select_time)

	#m_train_data, m_train_scaled_data, m_train_targets, m_valid_data, m_valid_scaled_data, m_valid_targets = get_train_validation_data(data, data_scaled, targets_selected)

	# given time steps, y is prepared 
	targets_selected_for_input = y_select_targets(targets_selected, select_time)

	# Baseline as random forest


	seed = 7
	np.random.seed(seed)
	"""rf_raw = RandomForestRegressor()
						
				param_dist_baseline = dict(max_depth=max_depth, n_estimators=n_estimators, bootstrap=bootstrap, min_samples_leaf = min_samples_leaf)
			
				baseline_random_search = RandomizedSearchCV(estimator = rf_raw, random_state = seed, param_distributions=param_dist_baseline, n_iter = models)
				y40 = []
				for y in targets:
					y40.append(y[-1])
				y40_list = [y.tolist() for y in y40]
				baseline_random_search.fit(targets_selected, y40_list)
			
				# summarize results
				print("Best: %f using %s" % (baseline_random_search.best_score_, baseline_random_search.best_params_))
				means = baseline_random_search.cv_results_['mean_test_score']
				stds = baseline_random_search.cv_results_['std_test_score']
				params = baseline_random_search.cv_results_['params']
				for mean, stdev, param in zip(means, stds, params):
					print("%f (%f) with: %r" % (mean, stdev, param))
				
				baseline_predictions = baseline_random_search.predict(targets_selected)
				mse_baseline = mean_squared_error(targets, baseline_predictions)
				plot_baseline_vs_true(targets, baseline_predictions,mse_baseline)
						"""
	# Last baseline

	rf_last = RandomForestRegressor()

	param_dist_baseline = dict(max_depth=max_depth, n_estimators=n_estimators, bootstrap=bootstrap, min_samples_leaf = min_samples_leaf)

	baseline_random_search_last = RandomizedSearchCV(estimator = rf_last, random_state = seed, param_distributions=param_dist_baseline, n_iter = models)

	inputs_last_baseline, targets_last_baseline = prepare_last_baseline(targets_original)
	targets_plot = y_select(targets_original, 5)
	targets_plot_4 = y_select_targets(targets_plot, 5)
	
	baseline_random_search_last.fit(inputs_last_baseline,targets_last_baseline)

	baseline_predictions = baseline_random_search_last.predict(inputs_last_baseline).reshape(265, 36)
	all_pred = np.c_[targets_plot_4,baseline_predictions]

	mse_baseline = mean_squared_error(targets, baseline_predictions[:,-1])

	mse_all =[]
	for z in range(data.shape[0]):
		mse_all.append(mean_squared_error(targets_original[z], all_pred[z]))

	plot_baseline_vs_true2(targets, baseline_predictions[:,-1],mse_baseline)
	#params = [[1e-4, 1e-7], 32, 10, 1e-6]	
	#plot_all_learning_curves_random(all_pred, targets_original, params, 5, mse_all)
	