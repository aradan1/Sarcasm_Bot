import time

import pandas as pd
import numpy as np
from pickle import dump, load # to save model so we dont waste 2-3 mins every time we restart
from os.path import join

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack

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

	print("\nDistribution of labeled samples:")
	print(df['label'].value_counts())


	print("\nHow labeled themes (subreddits) are:")
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

	# this will have the necessary tools to transform future data to fit into model
	model_dict = dict()

	# list containing the transformed data
	data =[]

	# build unigrams and bigrams, put a limit on maximal number of features on the vocabulary created
	# and minimal word frequency
	tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)

	# multinomial logistic regression (mostly just fixing default values in case of default shiffting)
	# also fixing the random_state to get results on same conditions in case of multiple runs
	# max_iter set really high since my pc is really "outdated"
	logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', random_state=10, verbose=1, max_iter=7600)

	# train
	print("\ntraining...")
	start = time.time()

	# Data 1 = comments
	data.append(tf_idf.fit_transform(x_train['comment']))
	model_dict["tf_idf"] = tf_idf

	# Data 2 = hasEmotes and numEmotes
	numeric = [i for i in x_train.columns if i in ["numEmotes","hasEmotes"]]
	data+=[x_train[i].values[np.newaxis].T for i in numeric]

	# Data 3 = Emote labels
	if "emotes" in x_train.columns:
		with open('faces.txt', 'r', encoding="utf-8") as f:
			matches = [i.replace('\n', '').replace('\r','') for i in f.readlines()]
		mlb = MultiLabelBinarizer(classes=matches)
		data.append(mlb.fit_transform(x_train['emotes']))
		model_dict["mlb"] = mlb

	X = hstack(data)

	logit.fit(X, y_train)

	end = time.time()
	print("time:",int((end - start)/60),"m",int(end - start)%60,"s")

	model_dict["logit"] = logit

	return model_dict

'''
	method that transforms data to fit the model

	data:= Df containing the data
	predictor:= Dictionary with the keys "tf_idf", "logit" and "mlb"(if required) after the correct fitting
'''
def transformData(predictor, data):
	result = []

	result.append(predictor["tf_idf"].transform(data["comment"]))

	numeric = [i for i in data.columns if i in ["numEmotes","hasEmotes"]]
	result += [data[i].values[np.newaxis].T for i in numeric]

	if "emotes" in data.columns:
		result.append(predictor["mlb"].transform(data['emotes']))

	return hstack(result)

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

	default_model_path = "dataset\\model\\"
	default_data_path = "data\\"


	paths = ["full_raw", "emote_checked", "emote_counted", "spellchecked", "tokenized_emote_counted", "lemmatized_tokenized_emote_counted"]
	dfs = {}
	for path in paths:
		df = pd.read_pickle(default_data_path+path+".pkl")
		#print(df.info())
		dfs[path]=df.copy()

	for name in dfs:
		print("\nDF: "+name+" ---------------------------")
		df=dfs[name]

		#dfinformation(df)
		# Get the sets for test and train
		x_train, x_test, y_train, y_test = train_test_split(df[df.columns[~df.columns.isin(['subreddit','author','label'])]], df['label'], random_state=10)
		
		predictor = modelFitting(x_train, y_train)

		#TO SAVE THE MODEL
		saveModel(default_model_path+name+"_model_2",predictor)

		# Check accuracy scores
		trans_x_test = transformData(predictor, x_test)
		prediction = predictor["logit"].predict(trans_x_test)
		modelInformation(prediction, y_test)