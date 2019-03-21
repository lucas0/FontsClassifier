import zipfile
import urllib.request
import sys
import random
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import os

FONTS_PATH = os.path.abspath('fonts')
DICT_PATH = os.path.abspath('idx_to_label')
filename = sys.argv[0]

cwd = os.path.abspath(filename+"/..")

def download_and_unzip():
	#download file
	download = "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip"
	print("Downloading Zip file into "+cwd+"/fonts.zip")
	with urllib.request.urlopen( download ) as url:
		#save
		output = open(cwd+"/fonts.zip", "wb")
		output.write(url.read())
		output.close()

	print("Unzipping file into folder "+cwd+"/fonts/")
	#unzips
	zip_ref = zipfile.ZipFile(cwd+"/fonts.zip", 'r')
	zip_ref.extractall(cwd+"/fonts/")
	zip_ref.close()

def y_to_one_hot(Y, vec_size):
	one_hot_vec = list()

	for y in Y:
		target = [0 for _ in range(vec_size)]
		target[y] = 1
		one_hot_vec.append(target)

	return np.array(one_hot_vec)

def plot_example(X):
	imgplot = plt.imshow(X[:,:,0])
	plt.show()

def get_dict(num_of_classes, fonts):
	if os.path.exists(DICT_PATH+str(num_of_classes)+".pickle"):
		pickle_in = open(DICT_PATH+str(num_of_classes)+".pickle","rb")
		idx_to_label = pickle.load(pickle_in)
		pickle_in.close()
		return idx_to_label
	else:
		idx_to_label = {idx:name for idx,name in enumerate(fonts)}

		pickle_out = open(DICT_PATH+str(num_of_classes)+".pickle","wb")
		pickle.dump(idx_to_label, pickle_out)	
		pickle_out.close()
		return idx_to_label

def data_load(split=0.7, filenames=["AGENCY"]):
	cwd = os.getcwd()
	fontsPath = cwd+"/fonts"
	
	filenames = list(filter(None, filenames))
	
	data = pd.concat([pd.read_csv(fontsPath+"/"+name+".csv") for name in filenames]).sample(frac=1)
	num_of_classes = len(data.font.unique())

	idx_to_label = get_dict(num_of_classes, filenames)
	label_to_idx = dict([[v,k] for k,v in idx_to_label.items()])

	X = data.iloc[:,12:].values
	Y = y_to_one_hot([label_to_idx[value] for value in data['font'].values], num_of_classes)

	X = np.true_divide(X,255)

	#next few commented lines print an example 
	rand = random.randint(1,len(data))
	print(rand)
	print(data.iloc[rand].font)
	print(Y[rand])

	splitpoint = int(math.floor(len(X)*split))
	X_train, X_test = X[:splitpoint], X[splitpoint:]
	Y_train, Y_test = Y[:splitpoint], Y[splitpoint:]

	X_train = np.reshape(X_train,(-1,20,20,1))
	X_test = np.reshape(X_test,(-1,20,20,1))

	return X_test,X_train,Y_test,Y_train,idx_to_label,label_to_idx
