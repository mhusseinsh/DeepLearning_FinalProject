import os
import sys
import glob
import json

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


def load_data(source_dir='./final_project'):
	
	configs = []
	learning_curves = []
	
	for fn in glob.glob(os.path.join(source_dir, "*.json")):
		with open(fn, 'r') as fh:
			tmp = json.load(fh)
			configs.append(tmp['config'])
			learning_curves.append(tmp['learning_curve'])
	return(configs, learning_curves)

def prepare_data():

	configs, learning_curves = load_data()

	Y_original = np.asarray(learning_curves)

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
	return X, Y, Y_original

def preprocess_data(X):
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	return X_scaled

def sort(X, X_scaled, y):
	y_sorted = np.zeros((y.shape[0],1))
	y_indices = []
	X_sorted = np.zeros((X.shape[0],5))
	X_scaled_sorted = np.zeros((X_scaled.shape[0],5))

	for i in range (len(y)):
		y_indices.append(y[i][0])

	indices = np.argsort(y_indices)[::-1]
	
	for i in indices:

		X_sorted[i] = X[i]
		y_sorted[i] = y[i] 
		X_scaled_sorted[i] = X_scaled[i] 
	
	return X_sorted, X_scaled_sorted, y_sorted

def get_train_validation_data(X, X_scaled, y):

	shuffled_index = np.random.permutation(len(y))
			
	indices_train = shuffled_index[0:int(0.9*len(y))]

	indices_valid = shuffled_index[int(0.9*len(y)):len(y)]

	#train_data = [X[i] for i in indices_train]
	train_data = X[indices_train]
	train_scaled_data = X_scaled[indices_train]
	train_targets = y[indices_train]

	valid_data = X[indices_valid]
	valid_scaled_data = X_scaled[indices_valid] 
	valid_targets = y[indices_valid]


	"""seed = 7
				np.random.seed(seed)
			
				train_data, valid_data, train_targets, valid_targets = train_test_split(X, y, test_size=0.1, random_state = seed)
				train_scaled_data, valid_scaled_data, train_targets, valid_targets = train_test_split(X, y, test_size=0.1, random_state = seed)
			"""
	return train_data, train_scaled_data,train_targets,valid_data,valid_scaled_data,valid_targets

def x_copy(X, time_steps):

	# reproduce x given sequence length
	x_copied = np.zeros((X.shape[0] * (time_steps - 1),X.shape[1]))

	count = 0
	for x in range(X.shape[0]):
		for t in range(time_steps-1):
			x_copied[count] = X[x]
			count += 1 

	return x_copied

def y_select(y_original, time_steps):

	# select y given sequence length
	y_selected = y_original.tolist()
	for row in y_selected:
		del row[time_steps:]

	y_selected = np.asarray(y_selected)
	return y_selected

def y_select_targets(y_selected, time_steps):

	# select y given sequence length
	y_out = np.zeros((y_selected.shape[0],time_steps-1))

	for y, i in zip(y_selected, range(y_selected.shape[0])):
		y_out[i] = np.delete(y, [time_steps-1])
	
	return y_out



def prepare_rnn_input(X, y_selected, time_steps):
	
	# gets copied x and concatenates it with y

	x_copied = x_copy(X, time_steps)
	x_y_rnn = np.zeros((x_copied.shape[0], x_copied.shape[1] + 1))

	i = 0
	count = 0

	y_flat = np.zeros((x_copied.shape[0],1))
	i = 0
	for y_s in y_selected:
		for y in y_s:
			if i == y_flat.shape[0]:
				break
			y_flat[i] = y
			i += 1

	x_y_rnn = np.c_[x_copied,y_flat]
	return x_y_rnn

def prepare_rnn_targets(y_selected, time_steps):

	# rnn targets given time sequence, first moment deleted 
	y_out = np.zeros((y_selected.shape[0],time_steps-1))

	for y, i in zip(y_selected, range(y_selected.shape[0])):
		y_out[i] = np.delete(y, [0])
	
	return y_out


def data_randomize(X, y_original):

	#randomly reproduce x and y
	input_lengths = [i for i in range(4,20)]
	random_lengths = np.random.choice(input_lengths, X.shape[0])
	

	# randomly reproduced x
	x_copied = np.repeat(X, random_lengths, axis=0)

	y_selected = y_original.tolist()
	for row, t in zip(y_selected, random_lengths):
		del row[t:]
	y_selected = np.array(y_selected)
	return x_copied, y_selected,random_lengths

def pad_targets_to_20(y_selected):

	y_selected = pad_sequences(y_selected, maxlen = 20, padding='post', dtype='float')

	return y_selected

def prepare_rnn_input_random(x_copied, y_selected):

	#random input for rnn
	
	y_flat = np.zeros((x_copied.shape[0],1))
	
	i = 0
	for y_s in y_selected:
		for y in y_s:
			y_flat[i] = y
			i += 1

	x_y_rnn = np.c_[x_copied,y_flat]

	#first_bucket = list(filter(lambda x: 5 <= len(x) <= 8 , x_y_rnn))
	#first_bucket = pad_sequences(first_bucket, maxlen = 8, padding='post', dtype='float')

	return x_y_rnn

def prepare_rnn_targets_random(y_selected, length):

	# rnn random targets
	y_out = []

	for y, i in zip(y_selected, range(y_selected.shape[0])):
		y_out.append(np.delete(y, [0]))

	y_flat = np.zeros((length,1))
	k = 0
	for y_s in y_out:
		#y_s_list = y_s.tolist()
		#for y in y_s_list[0]:
		for y in y_s:
			#print(k)
			y_flat[k] = y
			k += 1

	return np.array(y_flat)


def prepare_rnn_input_future(X, y_predicted):

	# given predicted y, prepare rnn input for future predictions

	x_y_rnn = np.zeros((X.shape[0],(X.shape[1] + 1)))

	count = 0

	for x, y in zip(X, y_predicted):
		x_y_rnn[count] = np.c_[np.array([x]),np.array([y])]
		count += 1

	return x_y_rnn

def prepare_last_baseline(y_original):

	y_input = []
	y_target = []
	for i in range(36):
		y_input.append(y_original.T[i : 4+i])

		y_target.append(y_original.T[4+i])


	return np.array(y_input).reshape(36*y_original.shape[0],4), np.array(y_target).reshape(36*y_original.shape[0],1)
















