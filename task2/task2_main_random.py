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
from rnn import *
from plotting import *


if __name__ == "__main__":


	decaying_lrs = [[1e-4, 1e-6], [1e-4, 1e-7], [1e-2, 1e-6], [1e-3, 1e-6]]
	alphas = [1e-7, 1e-6, 1e-5, 1e-4]
	models = 5
	pred_time = [5,10, 20, 30]
	train_time = [5,10,20]
	num_epochs = 1000

	# randomness
	kfold = get_folds()

	if not os.path.exists("./Plots/Train/Random"):
		os.makedirs("./Plots/Train/Random")
	
	# get data
	data, targets, targets_original = prepare_data()
	data_scaled = preprocess_data(data)

	data_randomly_replicated, targets_randomly_selected, random_lengths = data_randomize(data_scaled, targets_original)
	randomized_targets = targets_randomize(targets_original, random_lengths)

	# prepare data for cv 
	rnn_input = prepare_rnn_input_random(data_randomly_replicated, targets_randomly_selected)

	rnn_targets = prepare_rnn_targets_random(randomized_targets, data_scaled.shape[0],random_lengths)

	overall_mse = []
	best_weights = []
	lr_used = []
	alpha_used = []

	for model in range(models):
		# randomize hyperparams
		idx = np.random.randint(0, len(decaying_lrs))
		learningrate = decaying_lrs[idx]
		alpha = np.random.choice(alphas)
		lr_used.append(learningrate)
		alpha_used.append(alpha)
		# get data
		

		model = rnn(learning_rate=learningrate, num_epochs = num_epochs, alpha = alpha)
		for train, valid in kfold.split(rnn_input, rnn_targets):
			split_loss = []
			split_score = []
			split_mse = []

			all_split_score = []
			all_split_mse = []
			all_split_loss=[]

			
			for t_index in train:
				split_loss.append(model.fit(rnn_input[t_index].reshape(-1, random_lengths[t_index], 1 + data.shape[1]),
					rnn_targets[t_index].reshape(-1, random_lengths[t_index],1), epochs = num_epochs).history['loss'])

			all_split_loss.append(np.mean(split_loss))

			for v_index in valid:
				split_score.append(model.evaluate(rnn_input[v_index].reshape(-1, random_lengths[v_index], 1 + data.shape[1]), 
					rnn_targets[v_index].reshape(-1, random_lengths[v_index],1)) * 100)
				preds = model.predict(rnn_input[v_index].reshape(-1, random_lengths[v_index], 1 + data.shape[1]).reshape(-1, random_lengths[v_index], 1 + data.shape[1]))
			
				split_mse.append(mean_squared_error(preds[0], rnn_targets[v_index]))
				
			all_split_score.append(np.mean(split_score))
			all_split_mse.append(np.mean(split_mse))

		overall_score = np.mean(all_split_score)
		overall_mse.append(np.mean(all_split_mse))
		overall_loss = np.mean(all_split_loss)
		best_weights.append(model.get_weights())

	best = np.argmin(overall_mse)
	best_lr = lr_used[best]
	best_alpha = alpha_used[best]


	new_model = rnn_stateful(learning_rate=best_lr, num_epochs = num_epochs, alpha = best_alpha)
	new_model.set_weights(best_weights[best])
	
	overall_mse_test = []
	for s in pred_time:
		if not os.path.exists("./Plots/Train/Random/Test " + str(s)):
				os.makedirs("./Plots/Train/Random/Test " + str(s))
		targets_selected = y_select(targets_original, s)

		# given time steps, y is prepared 
		targets_selected_for_input = y_select_targets(targets_selected, s)
			
		# prepare data for cv
		rnn_input = prepare_rnn_input(data_scaled, targets_selected_for_input, s).reshape(-1, s-1, 1 + data.shape[1])
		

		all_preds = []

		for _, valid in kfold.split(rnn_input, rnn_targets):

			for sample_x in rnn_input[valid]:

				new_model.reset_states()
				prediction = 0
				preds = []

				for x in sample_x:
					prediction = new_model.predict(x.reshape(-1, 1, 1 + data.shape[1]), batch_size = 1)

				for i in range(s,41):
					x = sample_x[0][0:-1]
					x_pred = np.c_[np.array([x]),prediction[0]]
					prediction = new_model.predict(x_pred.reshape(-1, 1, 1 + data.shape[1]), batch_size = 1)

					preds.append(prediction[0])
				all_preds.append(np.array(preds).reshape(41-s,))
				
		predictions = np.c_[targets_selected_for_input.reshape(265,s-1),all_preds]
		
		params = [best_lr, best_alpha]

		thefile = open(os.path.join('./Plots/Train/Random' + '/', 'Predictions - Test_' + str(s) + '.txt'), 'w')
		for item in predictions:
		  thefile.write("%s\n" % item)

		mse_test = []
		for x, y in zip(predictions[:], targets_original[:]):
			mse_test.append(mean_squared_error(x[s-1:-1], y[s-1:-1]))

		mse_test = np.asarray(mse_test)
		overall_mse_test.append(mse_test)
		thefile = open(os.path.join('./Plots/Train/Random'+ '/', 'MSE - Test_' + str(s) + '.txt'), 'w')
		for item in mse_test:
		  thefile.write("%s\n" % item)

		plot_all_learning_curves_random(predictions, targets_original, params, s, overall_mse[best])
	plot_box_plots_random(np.asarray(overall_mse_test), params, s)

	thefile = open(os.path.join('./Plots/', 'targets' + '.txt'), 'w')
	for item in targets_original:
	  thefile.write("%s\n" % item)





