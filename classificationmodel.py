from utils import readpdf, removing_name_entities
from preparing_dataset import dfWordAsFeature
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing import df_manipulation

def LogResmodel(newX_train, newX_test, y_train, y_test):
	lrscaler = StandardScaler()
	# Fit on training set only.
	lrscaler.fit(newX_train.astype(float))
	# Apply transform to both the training set and the test set.
	X_train_sc = lrscaler.transform(newX_train.astype(float))
	X_test_sc = lrscaler.transform(newX_test.astype(float))
	numclass = len(set(y_train))
	print('There are', numclass, 'classes for this model')
	if numclass > 2:
		logresclass = 'multinomial'
	else:
		logresclass = 'ovr'
	lrmodel = LogisticRegression(solver = 'lbfgs', multi_class = logresclass)
	lrmodel.fit(X_train_sc, y_train)
	
	# use the model to make predictions with the train & test data
	y_train_pred = lrmodel.predict(X_train_sc)
	train_data_accuracy = metrics.accuracy_score(y_train, y_train_pred)
	
	y_test_pred = lrmodel.predict(X_test_sc)
	test_data_accuracy = metrics.accuracy_score(y_test, y_test_pred)
	
	#Saving Model and Standard Scaler
	pickle.dump(lrmodel, open('./static/model/lrmodel.sav', 'wb'))
	dump(lrscaler, './static/model/lrscaler.bin', compress=True)
	return(test_data_accuracy)

from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import zero_one_loss

def xgboostmodel(newX_train, newX_test, y_train, y_test):
	# META CODE	
	numclass = len(set(y_train))
	if numclass > 2:
		# You may need to use MultiLabelBinarizer to encode your variables from arrays [[x, y, z]] to a multilabel 
		# format before training.
		mlb = MultiLabelBinarizer()
		# Fit on training set only.
		mlb.fit(y_train)
		# Apply transform to both the training set and the test set.
		y_train_mlb = mlb.transform(y_train.astype('category'))
		y_test_mlb = mlb.transform(y_test.astype('category'))
		
		multixgb = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))
		multixgb.fit(newX_train.astype(float), y_train_mlb)
		
		pickle.dump(multixgb, open('./static/model/multixgb.sav', 'wb'))
		dump(mlb, './static/model/multixgbmlb.bin', compress=True)
		
		y_pred_gs_xgb = multixgb.predict(newX_test.astype(float))
		test_data_accuracy = 1 - zero_one_loss(y_test_mlb, y_pred_gs_xgb)
		return(test_data_accuracy)
	else:
		xgb = XGBClassifier(n_jobs=-1, max_depth=4)
		xgb.fit(newX_train.astype(float), y_train.astype('category'))
		
		pickle.dump(xgb, open('./static/model/xgb.sav', 'wb'))
		
		y_test_pred = xgb.predict(newX_test.astype(float))
		test_data_accuracy = metrics.accuracy_score(y_test.astype('category'), y_test_pred.astype('category'))
		return(test_data_accuracy)
	

import pickle
from sklearn.externals.joblib import dump, load
import pandas as pd
def testingpdf(path):	
	selectedword = pickle.load(open('./static/model/uniquewords.txt', 'rb'))
	ner = pickle.load(open('./static/model/detected_name_entities_file.txt', 'rb'))
	
	text = readpdf(path)
	#Convert text to inputdata(dataframe)
	#arrtext = removing_name_entities(text, ner)
	
	inputdata = pd.DataFrame(columns = ['RawText'])
	inputdata.loc[len(inputdata)] = text

	ner = pickle.load(open('./static/model/detected_name_entities_file.txt', 'rb'))
	processed_df = df_manipulation(inputdata, ner)
	newinputdata = dfWordAsFeature(processed_df, selectedword)
	
	#LogisticRegression
	lrmodel = pickle.load(open('./static/model/lrmodel.sav', 'rb'))
	lrscaler = load('./static/model/lrscaler.bin')
	#Scaler
	inputdata_sc = lrscaler.transform(newinputdata.astype(float))
	#Prediction
	logres_y_pred = lrmodel.predict(inputdata_sc)
	
	#Xgboost
	#multixgb = pickle.load(open('./static/model/multixgb.sav', 'rb'))
	#multixgbmlb = load('./static/model/multixgbmlb.bin')
	#Prediction
	#Mismatch in feature(column name)
	#xgb_y_pred = multixgb.predict(inputdata.astype(float))
	return(logres_y_pred)