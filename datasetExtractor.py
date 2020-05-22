import praw
from langdetect import detect, DetectorFactory, detect_langs

import pandas as pd
import numpy as np

import classifier

DetectorFactory.seed = 0


def getAllTaggedCommentsOfSubreddit(reddit, tag, subreddit, maxSeen=1000000, language="en"):

	dataset = {'comment':[], 
			'author':[],
			'subreddit':[]}
	sarcastis_found = 0
	try:
		while maxSeen > 0:

			submission = reddit.subreddit(subreddit).random()
			submission.comments.replace_more(limit=0)

			for comment in submission.comments.list():

				if maxSeen%1000==0:
					print(maxSeen)
				maxSeen -= 1

				if tag in comment.body:
					sarcastis_found += 1
					# get all comments either too short or in the requested language
					if len(comment.body.split()) == 1 or detect(comment.body) == language:
						dataset['comment'].append(comment.body)
						dataset['author'].append(comment.author.name)
						dataset['subreddit'].append(comment.subreddit.name)
				
	#except Exception as e:
	except:
		print("dies with "+str(maxSeen)+" comments left and "+str(len(dataset['comments']))+" comments added.")
		logging.error(traceback.format_exc())

	finally:
		print("found: "+str(sarcastis_found)+" comments")

		return dataset

# Checks all not tagged comments in a subreddit, defaults to the first 1 Million
def getAllNoneTaggedCommentsOfSubreddit(reddit, tag, subreddit, maxSeen=1000000, language="en"):

	dataset = {'comment':[], 
			'author':[],
			'subreddit':[]}
	try:
		while maxSeen > 0:

			submission = reddit.subreddit(subreddit).random()
			submission.comments.replace_more(limit=0)

			for comment in submission.comments.list():

				if maxSeen%100==0:
					print(maxSeen)
				maxSeen -= 1

				if tag not in comment.body:
					# get all comments either too short or in the requested language
					if len(comment.body.split()) == 1 or detect(comment.body) == language:
						dataset['comment'].append(comment.body)
						dataset['author'].append(comment.author.name)
						dataset['subreddit'].append(comment.subreddit.name)

	except:
		print("dies with "+str(maxSeen)+" comments left")
	finally:
		return dataset

#max comments checked per user is 1000, set by the api
def findNoneTaggedAuthorComments(author, tag, quantity):

	dataset = {'comment':[], 
			'subreddit':[]}

	for comment in reddit.redditor(author).comments.new(limit=None):
		if tag not in comment.body:
			dataset['comment'].append(comment.body)
			dataset['subreddit'].append(comment.subreddit.name)
			quantity -= 1

			if quantity == 0:
				break

	return dataset

#max comments checked per user is 1000, set by the api
def findTaggedAuthorComments(author, tag, quantity):

	dataset = {'comment':[], 
			'subreddit':[]}

	for comment in reddit.redditor(author).comments.new(limit=None):
		if tag in comment.body:
			dataset['comment'].append(comment.body)
			dataset['subreddit'].append(comment.subreddit.name)
			quantity -= 1

			if quantity == 0:
				break

	return dataset

# save dataframe as the specified filename
def saveToCSV(dataframe, filename):
	dataframe.to_csv(filename, index=False, encoding='utf-8')


if __name__ == '__main__':
	

	# This file must contain exactly 2 lines: client_id and client_secret
	with open("REDDIT_TOKEN") as myfile:
		client_id = next(myfile).strip()
		client_secret = next(myfile).strip()

	reddit = praw.Reddit(user_agent="Comment Extractor",
                     client_id=client_id, client_secret=client_secret)
	

	sarcasmTag = " /s"
	subreddit = "all"
	language = "es"
	maxSeen = 10000000 #10 Million

	df = pd.DataFrame( getAllTaggedCommentsOfSubreddit(reddit, sarcasmTag, subreddit, maxSeen=maxSeen,language=language) )
	
	print(df.shape)
	
	if df.shape[0] != 0:

		#TO SAVE THE DATAFRAME
		saveToCSV(df, "only_tagged.csv")
		df['sarcasm'] = 1
		print("found sarcastic comments")
		authorOcur = df['author'].value_counts()
		for author, quantity in authorOcur.items():

			# Find not number of not tagged comments from user
			nonTagged = findNoneTaggedAuthorComments(author, sarcasmTag, quantity)

			# make the dict fit the DataFrame
			lenght = len(nonTagged['comment'])
			nonTagged['author']=[author] * lenght
			nonTagged['sarcasm'] = [0] * lenght
			# append to DataFrame
			df = df.append(nonTagged, ignore_index=True)

		#TO SAVE THE DATAFRAME
		name = input("Name your file to be saved:\n")
		if not name.endswith(".csv"):
			name = name+".csv"
		saveToCSV(df, name)

