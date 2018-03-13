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


	decaying_lrs = [[1e-1, 1e-4],[1e-1, 1e-5],[1e-1, 1e-6],[1e-1, 1e-7],[1e-2, 1e-5],[1e-2, 1e-6],[1e-2, 1e-7],[1e-3, 1e-5],[1e-3, 1e-6],[1e-3, 1e-7],
			[1e-4, 1e-6],[1e-4, 1e-7],[1e-4, 1e-8],[1e-5, 1e-7]]
	batches = [8, 16, 32]
	alphas = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
	max_depth = [2,4,8,16,32]
	n_estimators = [4,8,16,32]
	bootstrap = [True, False]
	min_samples_leaf = [1,2,4,8]
	models = 30
	select_time = 5
	time_steps = [select_time]
	pred_time = [5,10, 20, 30]
	train_time = [5,10,20]
	models = 2
	num_epochs = 1


	# randomness
	seed = 7
	np.random.seed(seed)
	kfold = KFold(n_splits = 3, shuffle = True, random_state = seed)

	for l in train_time:
		if not os.path.exists("./Plots/Train " + str(l)):
			os.makedirs("./Plots/Train " + str(l))
		overall_mse = []
		data, targets, targets_original = prepare_data()
		data_scaled = preprocess_data(data)

		targets_selected = y_select(targets_original, l)

		# given time steps, y is prepared 
		targets_selected_for_input = y_select_targets(targets_selected, l)
			
		# prepare data for cv
		rnn_input = prepare_rnn_input(data_scaled, targets_selected_for_input, l).reshape(-1, l-1, 1 + data.shape[1])

		rnn_targets = prepare_rnn_targets(targets_selected, l).reshape(-1, l-1,1)

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

				for x, y in zip(rnn_input[train],rnn_targets[train]):
					split_loss.append(model.fit(x.reshape(-1, l-1, 1 + data.shape[1]), y.reshape(-1, l-1,1), epochs = num_epochs).history['loss'])

				all_split_loss.append(np.mean(split_loss))

				for x, y in zip(rnn_input[valid],rnn_targets[valid]):
					split_score.append(model.evaluate(x.reshape(-1, l-1, 1 + data.shape[1]), y.reshape(-1, l-1,1)) * 100)
					preds = model.predict(x.reshape(-1, l-1, 1 + data.shape[1]))
				
					split_mse.append(mean_squared_error(preds[0], y))

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
		
		for s in pred_time:
			if not os.path.exists("./Plots/Train " + str(l) + "/Test " + str(s)):
				os.makedirs("./Plots/Train " + str(l) + "/Test " + str(s))
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
			plot_all_learning_curves(predictions, targets_original, params, l, s, overall_mse[best])






