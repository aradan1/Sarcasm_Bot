from langdetect import detect, DetectorFactory

import spacy

import pandas as pd

import requests
from datetime import datetime
import time
import re

import classifier

#DetectorFactory.seed = 0
nlp = spacy.load('es_core_news_sm')

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


'''
	Removes any consecutive repetition of "char" in "string"
	(ex: 'a' and "aaaia" would return "aia") 
'''
def discardRepetition(string, char): 
    pattern = char + '{2,}'
    string = re.sub(pattern, char, string) 
    return string


'''
	Removes the substring occurrences inside string and returns the number
	of times that happened

	string:= String to be searched into
	sub:= substring to look for
'''
def removeAndCountSub(string, sub):
	long1=len(string)
	result= string.replace(sub,'')
	long2=len(result)
	if long1==long2:
		return result, 0
	return result, (long1-long2)/len(sub)

'''
	Lemmatizes string

	string:= Text to process
'''
def lemmatize(string):
	doc = nlp(string)
	tokens = [word for word in doc if not word.is_punct]
	lemmas = [tok.lemma_.strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
	return " ".join(lemmas)

'''
	Converts string to dictionary that fits the Lemmatized_Tokenized_Emote_Counted dataframe

	string:= comment to convert
'''
def convertToLTEC(string):
	df = convertToECounted(string)
	for index, row in df.iterrows():
		df.at[index,"comment"] = lemmatize(row["comment"])
	return df


'''
	Converts string to dictionary that fits the Emote_Counted dataframe

	string:= comment to convert
'''
def convertToECounted(string):
	if string.endswith("."):
		string=string[:-1]
	with open('faces.txt', 'r', encoding="utf-8") as f:
		matches = [i.replace('\n', '').replace('\r','') for i in f.readlines()]
	count = 0
	emotes = []
	for face in matches:
		if " "+face+" " in " "+string+" ":
			string, num = removeAndCountSub(string, face)
			count+=num
			emotes.append(face)

	return pd.DataFrame({"comment":[string.lower()], "emotes":[emotes], "numEmotes":[count]})

'''
	Converts string to dictionary that fits the Emote_Checked dataframe

	string:= comment to convert
'''
def convertToEChecked(string):
	if string.endswith("."):
		string=string[:-1]
	with open('faces.txt', 'r', encoding="utf-8") as f:
		matches = [i.replace('\n', '').replace('\r','') for i in f.readlines()]
	isIn = [" "+face+" " in " "+string+" " for face in matches]

	return pd.DataFrame({"comment": [string], "hasEmotes": [any(isIn)], "numEmotes": [isIn.count(True)]})


if __name__ == '__main__':


	default_backup_path = "backup\\"
	default_dataset_path = "dataset\\"
	default_data_path = "data\\"
	sarcasmTag = " /s"
	language = "es"
	maxSeen = 1e6