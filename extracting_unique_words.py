import pandas as pd
from gensim.models import TfidfModel
from gensim import corpora

#ExtractedKeywords
def getcommonnessword(dataframe, docindex = pd.Series(data = None)):
	if docindex.any():
		dataframe = dataframe.loc[dataframe.index.isin(docindex)]
	tokens = list(dataframe['tokens3'])
	#Get Corpus & Dictionary
	dictionary = corpora.Dictionary(tokens)
	#dictionary_slctdtopic.filter_extremes(no_below=4, no_above=0.6)
	corpus = [dictionary.doc2bow(tok) for tok in tokens]
	
	# Create the TF-IDF model
	tfidf_slctdtopic_model = TfidfModel(corpus, smartirs='ntc')
	corpus_tfidf = tfidf_slctdtopic_model[corpus]
	dict_tfidf = {dictionary.get(id): value for doc in corpus_tfidf for id, value in doc}
	
	#Sort the value from the most common to 
	sort_dict = sorted(dict_tfidf.items(), key = lambda x : x[1])
	return(sort_dict)

from kneed import KneeLocator
def getmostcommonword(dataframe, docindex = pd.Series(data = None)):
	wordcommoness = getcommonnessword(dataframe, docindex)
	docvalue = []
	docerrorvalue = []
	for i in range(0, len(wordcommoness)-1):
		docvalue.append(wordcommoness[i][1])
		docerrorvalue.append(wordcommoness[i+1][1] - wordcommoness[i][1])
	basenum = int(len(docvalue)*0.1)
	x = range(0, len(docvalue))
	kn = KneeLocator(x[basenum:-1000], docvalue[basenum:-1000], curve='concave', direction='increasing')
	
	commonworddoc = []
	for i in range(0, kn.knee):
		commonworddoc.append(wordcommoness[i][0])
	return(commonworddoc)
	
def slicingCW(commonword, specificcommonword):
	copy = specificcommonword.copy()
	for word in commonword:
		if word in copy:
			copy.remove(word)
	return(copy)

import pickle
def getfeaturedword(X_train, y_train):
	#Common word for input files
	commonwordwholedoc = getmostcommonword(X_train)

	targetname = set(y_train)
	topicindex = []
	for tname in targetname:
		topicindex.append(y_train.index[y_train == tname])

	containerCW = []
	#Topics contain index of X_train for each topic
	for ind in topicindex:
		containerCW.append(getmostcommonword(X_train, ind))

	slicedword = []
	for cw in  containerCW:
		slicedword.append(slicingCW(commonwordwholedoc, cw))

	pureslicedword = []
	for i in range(0,len(slicedword)):
		x = slicedword.copy()
		del x[i]
		topic = list(set(x[0] + x[1]))
		#print(slicingCW(topic, slicedword[i]))
		pureslicedword = pureslicedword + slicingCW(topic, slicedword[i])
	pureslicedword.sort()
	#Saving SelectedWord
	pickle.dump(pureslicedword, open('./static/model/uniquewords.txt', 'wb'))
	return(pureslicedword)