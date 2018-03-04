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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam

def load_data(source_dir='./../final_project'):
	
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

X, y, y_original = prepare_data(configs, learning_curves)
X_scaled = preprocess_data(X)

def mlp(X, y, batch_size, num_epochs, learning_rate, raw):

	model = Sequential()
	"""model.add(Dense(64, input_dim = 5, kernel_initializer = 'random_uniform', 
					bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
				#model.add(Dropout(0.2))
				model.add(Dense(64, kernel_initializer = 'random_uniform', 
					bias_initializer = 'zeros', activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
				#model.add(Dropout(0.2))
				model.add(Dense(1, kernel_initializer = 'random_uniform', kernel_regularizer=regularizers.l2(0.01)))"""

	model.add(Dense(64, input_dim = 5, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu'))
	#model.add(Dropout(0.1))
	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = 'zeros', activation = 'relu'))
	#model.add(Dropout(0.1))
	model.add(Dense(1, kernel_initializer = 'random_uniform'))

	#decay = learning_rate / num_epochs

	#sgd = SGD(lr=0.1, decay=0.0, momentum=0.9)
	adam = Adam(lr=1e-5)
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	print('start training')

	def exponential_decay(epoch):
			
		return np.exp(-epoch/5000)

	early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto')
	#l_rate = LearningRateScheduler(exponential_decay)

	callback_list =[early_stopping]

	history = model.fit(X, y, epochs = num_epochs, batch_size = batch_size, validation_split = 0.1, shuffle = True, callbacks = callback_list)
	print('training done')
	if raw == True:
		model_json = model.to_json()
		with open("model_raw.json", "w") as json_file:
			json_file.write(model_json)

		model.save_weights("model_raw.h5")
		print("raw model is saved")
	else:
		model_json = model.to_json()
		with open("model_pre.json", "w") as json_file:
			json_file.write(model_json)

		model.save_weights("model_pre.h5")
		print("preprocessed model is saved")
	return history


"""def baseline_mlp():
	print("In the baseline...")
	model = Sequential()
	model.add(Dense(5, input_dim = 5, kernel_initializer = 'normal', activation = 'relu'))
	#model.add(Dense(20, kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model

def baseline_linear(X, y):
	bias = np.ones((X.shape[0],1))
	X_new = np.c_[bias, X]
	w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.t, y))
	#w = np.linalg.inv(X_new.T.dot(X)).dot(X.T.dot(y))
	y_hat = X_new.dot(w)
	return y_hat

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
	print("Results (without pipeline): %.2f (%.2f) MSE" % (results.mean(), results.std()))"""

def test(raw):
	if (raw):
		json_file = open('model_raw.json', 'r')
		model = json_file.read()
		json_file.close()
		ml = model_from_json(model)
		ml.load_weights("model_raw.h5")
		print("Raw Model Loaded")
		y_error = []
		for x_test, y_test in zip(X, y):
			x_test = x_test.reshape(-1, 5)
			y_pred = ml.predict(x_test)
			y_diff = abs(y_pred - y_test)
			y_error.append(y_diff)
		y_mean_error = np.mean(y_error)
		print("mean accuracy from raw data", y_mean_error)

	else:
		json_file = open('model_pre.json', 'r')
		model = json_file.read()
		json_file.close()
		ml = model_from_json(model)
		ml.load_weights("model_pre.h5")
		print("Pre Model Loaded")
		y_error = []
		for x_test, y_test in zip(X_scaled, y):
			x_test = x_test.reshape(-1, 5)
			y_pred = ml.predict(x_test)
			y_diff = abs(y_pred - y_test)
			y_error.append(y_diff)
		y_mean_error = np.mean(y_error)
		print("mean accuracy from preprocessed data", y_mean_error)

def evaluate(X, raw):
	if (raw):
		json_file = open('model_raw.json', 'r')
		model = json_file.read()
		json_file.close()
		ml = model_from_json(model)
		ml.load_weights("model_raw.h5")
		print("Raw Model Loaded")
		y_net = []
		for x_test, y_test in zip(X, y):
			x_test = x_test.reshape(-1, 5)
			y_pred = ml.predict(x_test)
			y_net.append(y_pred[0])
		return y_net

	else:
		json_file = open('model_pre.json', 'r')
		model = json_file.read()
		json_file.close()
		ml = model_from_json(model)
		ml.load_weights("model_pre.h5")
		print("Pre Model Loaded")
		y_net = []
		for x_test, y_test in zip(X, y):
			x_test = x_test.reshape(-1, 5)
			y_pred = ml.predict(x_test)
			y_net.append(y_pred[0])
		return y_net

Training = True
batch = 64
epochs = 1500
if (Training):
	history = mlp(X, y, batch, epochs, 0.1, raw = True)
	fig1 = plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss (raw data)', fontsize=20, fontweight="bold")
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='best', fancybox=True, framealpha=0.5)
	fig1.savefig('model_loss_raw.png')

	history_scaled = mlp(X_scaled, y, batch, epochs, 0.1, raw = False)
	fig2 = plt.figure()
	plt.plot(history_scaled.history['loss'])
	plt.plot(history_scaled.history['val_loss'])
	plt.title('Model Loss (scaled data)', fontsize=20, fontweight="bold")
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='best', fancybox=True, framealpha=0.5)
	fig2.savefig('model_loss_scaled.png')
	raw = False
	test(raw)
	raw = True
	test(raw)
else:
	raw = False
	test(raw)
	raw = True
	test(raw)

#y = sorted(y, reverse=True)
y_sorted = []
for i in range (len(y)):
	y_sorted.append(y[i][0])

indices = np.argsort(y_sorted)[::-1]

X_sorted = [X[i] for i in indices]
y_sorted = [y[i] for i in indices]
X_scaled_sorted = [X_scaled[i] for i in indices]

# RAW DATA EVALUATION
# Create linear regression object
lr = linear_model.LinearRegression()

# Predict using the fitted model (raw data)
y_pred = cross_val_predict(lr, X_sorted, y_sorted, cv=3)

# Predict using the network (raw data)
raw = True
y_net = evaluate(X_sorted, raw)

print("using baseline to compare: ")
# The difference between the mlp and the baseline
y_error = abs(y_pred - y_net)
print("baseline scores predict raw data: ", np.mean(y_error))

# True vs Baseline
fig, ax = plt.subplots()
ax.scatter(y_sorted, y_pred, edgecolors=(0, 0, 0))
ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
ax.set_xlabel('True Values')
ax.set_ylabel('Baseline Values')
ax.set_title('True vs Baseline (raw data)', fontsize=20, fontweight="bold")
#plt.show()
plt.savefig('model_rawData(baseline).png')

#True vs Network
fig, ax = plt.subplots()
ax.scatter(y_sorted, y_net, edgecolors=(0, 0, 0))
ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
ax.set_xlabel('True Values')
ax.set_ylabel('Network Values')
ax.set_title('True vs Network (raw data)', fontsize=20, fontweight="bold")
#plt.show()
plt.savefig('model_rawData(network).png')

fig3 = plt.figure()
plt.plot(y_sorted)
plt.plot(y_pred)
plt.plot(y_net)
plt.title('Comparison (raw data)', fontsize=20, fontweight="bold")
plt.ylabel('y Value')
plt.xlabel('Samples')
l2norm = np.linalg.norm(y_sorted - np.array(y_net))
plt.legend(['y_true', 'y_baseline', 'y_network ,' + ' ,L2 Norm =' + str(l2norm)], loc='best', fancybox=True, framealpha=0.5)
fig3.savefig('metrics_comparison_raw.png')


# SCALED DATA EVALUATION
# Create linear regression object
lr = linear_model.LinearRegression()

# Predict using the fitted model (raw data)
y_pred = cross_val_predict(lr, X_scaled_sorted, y_sorted, cv=3)

# Predict using the network (raw data)
raw = False
y_net = evaluate(X_scaled_sorted, raw)

print("using baseline to compare: ")
# The difference between the mlp and the baseline
y_error = abs(y_pred - y_net)
print("baseline scores predict scaled data: ", np.mean(y_error))

# True vs Baseline
fig, ax = plt.subplots()
ax.scatter(y_sorted, y_pred, edgecolors=(0, 0, 0))
ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
ax.set_xlabel('True Values')
ax.set_ylabel('Baseline Values')
ax.set_title('True vs Baseline (scaled data)', fontsize=20, fontweight="bold")
#plt.show()
plt.savefig('model_scaledData(baseline).png')

#True vs Network
fig, ax = plt.subplots()
ax.scatter(y_sorted, y_net, edgecolors=(0, 0, 0))
ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
ax.set_xlabel('True Values')
ax.set_ylabel('Network Values')
ax.set_title('True vs Network (scaled data)', fontsize=20, fontweight="bold")
#plt.show()
plt.savefig('model_scaledData(network).png')


fig4 = plt.figure()
plt.plot(y_sorted)
plt.plot(y_pred)
plt.plot(y_net)
plt.title('Comparison (scaled data)', fontsize=20, fontweight="bold")
plt.ylabel('y Value')
plt.xlabel('Samples')
l2norm = np.linalg.norm(y_sorted - np.array(y_net))
plt.legend(['y_true', 'y_baseline', 'y_network ,' + ' ,L2 Norm =' + str(l2norm)], loc='best', fancybox=True, framealpha=0.5)
fig4.savefig('metrics_comparison_scaled.png')
#print(history)
exit()
"""evaluate_raw_data(X, y, baseline)
evaluate_preprocessed_data_pipeline(X, y, baseline)
evaluate_preprocessed_data(X, y, baseline)"""