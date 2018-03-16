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

from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import *
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import Adam
from keras.regularizers import l1,l2
from keras.initializers import Constant

def mlp(num_epochs, learning_rate, alpha):
	model = Sequential()

	model.add(Dense(64, input_dim = 5, kernel_initializer = 'random_uniform', 
		bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l2(alpha)))
		#bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l1(alpha)))
	#model.add(GaussianNoise(0.02))

	model.add(Dense(64, kernel_initializer = 'random_uniform', 
		bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l2(alpha)))
		#bias_initializer = Constant(0.1), activation = 'relu', kernel_regularizer=l1(alpha)))
	#model.add(GaussianNoise(0.02))
	model.add(Dense(1, kernel_initializer = 'random_uniform'))

	#initial_lr = learning_rate[0]
	#final_lr = learning_rate[1]

	#decay_factor = (initial_lr - final_lr)/num_epochs
	
	#adam = Adam(lr=learning_rate, decay = decay_factor)
	adam = Adam(lr=learning_rate)
	
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')

	return model

