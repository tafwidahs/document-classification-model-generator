import pandas as pd

import time

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from preprocessing import df_manipulation

def gettraintest(filename, data, label):
	new_labels = ['FileName', 'RawText', 'Target']
	oridf = pd.DataFrame(list(zip(filename, data, label)), columns=new_labels)

	print('-------------Start Manipulating Data-------------')
	df = df_manipulation(oridf)
	print('-------------------------------------------------')

	df = df.sample(frac=1).copy()
	print(df['Target'].value_counts())
	
	X = df.drop(columns = {'Target'})
	y = df['Target'].copy()
	skf = StratifiedKFold(n_splits=5)

	for train_index, test_index in skf.split(X, y):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	print(y_train.value_counts())
	print(y_test.value_counts())
	return(X_train, X_test, y_train, y_test)

def dfWordAsFeature(dataframe, words):
	dictforword = {}
	finaldf = pd.DataFrame(columns = words)
	for w in words:
		dictforword[w] = 0
		
	for row in dataframe['tokens3']:
		#Processing a row
		dictforword = dictforword.fromkeys(dictforword, 0)
		for token in row:
			if token in dictforword:
				dictforword[token] = dictforword[token] + 1
		buff = pd.DataFrame([dictforword])
		finaldf = pd.concat([finaldf, buff], ignore_index = True)
	return(finaldf)