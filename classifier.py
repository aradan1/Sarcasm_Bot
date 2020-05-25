from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

import time

import pandas as pd
import numpy as np
from pickle import dump, load # to save model so we dont waste 2-3 mins every time we restart
from os.path import join

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# heatmap plots
import seaborn as sn
import matplotlib.pyplot as plt
import eli5



'''
	Prints some information about the DataFrame

	df:= DataFrame from which information will be obtained
'''
def dfinformation(df):

	print("\nshape:")
	print(df.shape)

	print("\nInfo on df:")
	print(df.info())

	print("\nDistribution of sarcastic/non-sarcastic samples:")
	print(df['label'].value_counts())


	print("\nHow sarcastic themes (subreddits) are:")
	# this seemed usefull since subreddits appeal to topics and in the future information like this could be usefull
	sub_df = df.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum])
	print(sub_df[sub_df['size'] > 1000].sort_values(by='mean', ascending=False).head(10))
	# print(sub_df.sort_values(by='sum', ascending=False).head(10))


'''
	Creates a pipeline with a vectorizer and Log.Regression and makes a 'fit' with the train sets

	x_train:= train question
	y_train:= train response
'''
def modelFitting(x_train, y_train):
	# build unigrams and bigrams, put a limit on maximal number of features on the vocabulary created
	# and minimal word frequency
	tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)

	# multinomial logistic regression (mostly just fixing default values in case of default shiffting)
	# also fixing the random_state to get results on same conditions in case of multiple runs
	# max_iter set really high since my pc is really "outdated"
	logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', random_state=10, verbose=1, max_iter=7600)

	# sklearn's pipeline
	tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), ('logit', logit)])

	# train
	print("\ntraining...")
	start = time.time()
	tfidf_logit_pipeline.fit(x_train, y_train)
	end = time.time()
	print("time:",int((end - start)/60),"m",int(end - start)%60,"s")

	return tfidf_logit_pipeline

'''
	Prints model information like accuracy scores and a confusion matrix

	prediction:= Predicted response from the model
	y_test:= Correct response
	plotSize:= Size of the ploted confusion matrix
'''
def modelInformation(prediction, y_test, plotSize = (4,4)):

	print("\nAccuracy of the model:")
	print(accuracy_score(y_test, prediction))

	cm = confusion_matrix(y_test, prediction)
	cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize = plotSize)
	sn.heatmap(cmn, annot=True)
	plt.ylabel('Predicted label')
	plt.xlabel('True label')
	
	plt.show()

'''
	Saves data in file

	path:= Path to the file in which we want data to be stored
	data:= Data to be stored
'''
def saveModel(path, data):
	if not path.endswith(".pkl"):
		path = path+".pkl"
	with open(path, 'wb') as f:
		dump(data, f)

'''
	Loads data from file

	path:= Path from which we want to recover data
'''
def loadModel(path):
	if not path.endswith(".pkl"):
		path = path+".pkl"

	with open(path, 'rb') as f:
		data = load(f)
	return data



if __name__ == '__main__':

	try:
		Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
		path = askopenfilename()
		predictor = loadModel(path)

	except IOError:
		

		Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
		path = askopenfilename()

		df = pd.read_csv(path)	
		# noticed df.info() returned some Null values under the label comments, can't have those
		df.dropna(subset=['comment'], inplace=True)

		dfinformation(df)
		# Get the sets for test and train
		x_train, x_test, y_train, y_test = train_test_split(df['comment'], df['label'], random_state=10)

		predictor = modelFitting(x_train, y_train)

		#TO SAVE THE MODEL
		Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
		path = askdirectory()
		name = input("Name your file to be saved:\n")
		if not path.endswith("\\"):
			path = path+"\\"
		saveModel(path+name,predictor)

		# Check accuracy scores
		prediction = predictor.predict(x_test)
		modelInformation(predictor, prediction, y_test)


	print("\n\nTesting examples:\n")
	while True:
		statement = input("What would you like to say? (type '1' to exit)\n")
		if statement == "1":
			break
		print("Is this sarcastic?",predictor.predict([statement])[0] == 1)


