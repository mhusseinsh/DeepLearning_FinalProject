import os
import sys
import glob
import json

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
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam

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

def y_select(time_steps):
	y_selected = y_original.tolist()
	for row in y_selected:
		del row[time_steps:]

	y_selected = np.asarray(y_selected)
	return y_selected

def preprocess_data(X):
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	return X_scaled

def x_copy(X, time_steps):
	x_copied = np.zeros((X.shape[0] * time_steps,X.shape[1])).reshape(X.shape[0], time_steps, X.shape[1])

	for x in range(X.shape[0]):
		for t in range(time_steps):
			x_copied[x][t] = X[x]

	return x_copied

def sort(X, X_scaled, y):
	y_sorted = np.zeros((265,1))
	y_indices = []
	X_sorted = np.zeros((265,5))
	X_scaled_sorted = np.zeros((265,5))

	for i in range (len(y)):
		y_indices.append(y[i][0])

	indices = np.argsort(y_indices)[::-1]
	
	for i in indices:

		X_sorted[i] = X[i]
		y_sorted[i] = y[i] 
		X_scaled_sorted[i] = X_scaled[i] 
	
	return X_sorted, X_scaled_sorted, y_sorted

def get_train_validation_data(X, X_scaled, y):
	"""shuffled_index = np.random.permutation(len(y))
			
				indices_train = shuffled_index[0:int(0.9*len(y))]
				indices_valid = shuffled_index[int(0.9*len(y)):len(y)]
			
				#train_data = [data[i] for i in indices_train]
				train_data = X[indices_train]
				train_scaled_data = X_scaled[indices_train]
				train_targets = y[indices_train]
			
			
				valid_data = X[indices_valid]
				valid_scaled_data = X_scaled[indices_valid] 
				valid_targets = y[indices_valid] """

	seed = 7
	np.random.seed(seed)

	train_data, valid_data, train_targets, valid_targets = train_test_split(X, y, test_size=0.1, random_state = seed)
	train_scaled_data, valid_scaled_data, train_targets, valid_targets = train_test_split(X, y, test_size=0.1, random_state = seed)

	return train_data, train_scaled_data,train_targets,valid_data,valid_scaled_data,valid_targets

