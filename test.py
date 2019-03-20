import pandas as pd
import math
import numpy as np
import os
import collections
import h5py
from utils import data_load, plot_example
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model, load_model
from keras.initializers import Initializer, RandomNormal, RandomUniform
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.constraints import maxnorm
from keras.regularizers import l2

cwd = os.getcwd()
fontsPath = cwd+"/fonts"

fonts = []
for root, dirs, files in os.walk(fontsPath):
	for e in files:	
		fonts.append(e.split(".")[0])

# fonts = ['PALATINO', 'STYLUS', 'NINA', 'GOUDY']

X_test,X_train,Y_test,Y_train,idx_to_label,label_to_idx = data_load(0.8,fonts)

def get_model(X_train, Y_train, target_shape=153):
	model_name = "model_with_target_"+str(target_shape)+".h5"

	if os.path.exists(model_name):
		return load_model(model_name)

	else:
		X_input = Input(shape=(20,20,1,))

		conv = Conv2D(32, (4, 4), activation='relu', padding='same', kernel_regularizer=l2(0.01))(X_input)

		pool = MaxPooling2D(pool_size=(2, 2))(conv)

		conv = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(X_input)

		pool = MaxPooling2D(pool_size=(2, 2))(conv)

		#dropout = Dropout(0.5)(pool)

		flat = Flatten()(dropout)

		dense1 = Dense(512, activation='relu', kernel_constraint=maxnorm(3), kernel_regularizer=l2(0.01))(flat)

		dropout = Dropout(0.5)(dense1)

		dense2 = Dense(target_shape, activation='softmax', kernel_regularizer=l2(0.01))(dropout)

		model = Model(inputs=X_input, outputs=dense2)

		#model.compile(loss = 'categorical_crossentropy', optimizer=Adagrad(lr=0.01, epsilon=None, decay=0.0), metrics=['accuracy'])
		model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		# Save the model
		model.fit(X_train, Y_train, epochs=10, batch_size=32)
		model.save(model_name)
		return(model)


model = get_model(X_train, Y_train, len(idx_to_label))
score = model.evaluate(X_test, Y_test)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
