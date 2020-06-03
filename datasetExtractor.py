from langdetect import detect, DetectorFactory, detect_langs
from spellchecker import SpellChecker


import pandas as pd
import numpy as np

import requests
from datetime import datetime
import traceback
import time
import signal
import sys
import os

import classifier

DetectorFactory.seed = 0

url = "https://api.pushshift.io/reddit/comment/search?limit=1000&sort=desc&{}={}&before="
default_backup_path = ""
default_dataset_path = ""


'''
	Save pandas.DataFrame as csv file

	dataframe:= DataFrame to be saved
	filename:= Path wanted for the csv file 
'''
def saveDFToCSV(dataframe, filename):
	dataframe.to_csv(filename, index=False, encoding='utf-8')


''' 
	Searches all comments from reddit with specified pattern. 
	(API documentation: https://pushshift.io/api-parameters/)

	pattern:= Field with which will search comments (endpoint from the api. ex: subreddit, author).
	target:= Value to search 'pattern' for.
	tag:= String to look for in the comments("" will result in all comments matched).
	tagged:= Return only comments that: 'True' = contain the tag, 'False' = don't contain the tag.
	limit:= Number of comments wanted to check (if 'None' check all posible comments, if negative none will be checked)
	quantity:= Max number of matches expected to save (if negative this condition will be ignored) 
'''
def downloadFromUrl(pattern, target, tag, tagged, limit=None, language="es" ,quantity=-1):
	print(f"Saving comments from {pattern} {target}")
	maxSeen = 0
	count = 0
	dataset = {'comment':[], 
			'author':[],
			'subreddit':[]}

	start_time = datetime.utcnow()
	previous_epoch = int(start_time.timestamp())

	def signal_handler(sig, frame):
		print("Saving current data before exiting")
		classifier.saveModel(default_backup_path+target, (previous_epoch, dataset))
		sys.exit(0)

	signal.signal(signal.SIGINT, signal_handler)

	while limit==None or maxSeen < limit:
		new_url = url.format(pattern,target)+str(previous_epoch)
		json = requests.get(new_url, headers={'User-Agent': "Comment downloader"})
		time.sleep(1) # pushshift has a rate limit, if we send requests too fast it will start returning error messages
		try:
			json_data = json.json()
		except ValueError:
			print("Response content is not valid JSON")
			continue

		if 'data' not in json_data:
			break
		objects = json_data['data']

		if len(objects) == 0 or count == quantity:
			break

		for object in objects:

			if count == quantity or (limit!=None and maxSeen == limit):
				break

			#if maxSeen is a number
			if maxSeen%1000==0:
				print(maxSeen)
			maxSeen += 1

			previous_epoch = object['created_utc'] - 1
			
			try:
				text = object['body']
				textASCII = text.encode(encoding='ascii', errors='ignore').decode()
				if (tag in textASCII) == tagged:
					if pattern != "author" or detect(textASCII)==language:
						count += 1
						if tagged:
							textASCII = trimTaggedString(textASCII, tag)
						dataset['comment'].append(textASCII)
						dataset['author'].append(object['author'])
						dataset['subreddit'].append(object['subreddit'])

			except Exception as err:
				print(f"Couldn't print comment")
				continue
	
	print(f"Saved {count} comments out of {maxSeen} checked from {target}")
	classifier.saveModel(default_dataset_path+pattern+"\\"+target, dataset)
	return dataset

'''
	Merges all dictionaries inside a list into a single dictionary

	dict_list:= List of dictionaries to merge
'''
def mergeDictionaries(dict_list):
	result = {'comment':[], 
			'author':[],
			'subreddit':[]}
	for d in dict_list:
		result['comment'] = result['comment'] + d['comment']
		result['author'] = result['author'] + d['author']
		result['subreddit'] = result['subreddit'] + d['subreddit']

	return result

'''
	Splits the string from where the tag is found and removes the later part

	string:= String wanted to trim
	tag:= Substring to look for in 'string'
'''
def trimTaggedString(string, tag):
	return string.split(tag)[0].strip()




if __name__ == '__main__':

	default_backup_path = "backup\\"
	default_dataset_path = "dataset\\"
	sarcasmTag = " /s"
	language = "es"
	maxSeen = 1e6

	df = pd.read_csv("\\data\\full_raw.csv")