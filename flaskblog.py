from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename
import os
import time
import csv
import pandas as pd
import unicodedata

from utils import extracttext
from preparing_dataset import gettraintest, dfWordAsFeature
from extracting_unique_words import getfeaturedword
from classificationmodel import LogResmodel, xgboostmodel, testingpdf

FLOw_FOLDER = os.path.join('static', 'flowimage')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FLOw_FOLDER

uploadfiles = []

@app.route("/")
@app.route("/home")
def home():
	full_path = os.path.join(app.config['UPLOAD_FOLDER'], '1-half.png')
	return render_template('home.html', flow_image = full_path)

@app.route('/process',methods=["POST"])
def process():
	full_path = os.path.join(app.config['UPLOAD_FOLDER'], '2-half.png')
	if request.method == 'POST':
		#Read NER names list


		#Unzip & Process per folder
		#Path to Files
		zipfilespath = "./static/uploadstorage"

		print('Start text extraction from PDFs')
		filename, data, label = extracttext(zipfilespath)
		#Get Train & Test data
		X_train, X_test, y_train, y_test = gettraintest(filename, data, label)
		
		#Slicing Words
		selectedword = getfeaturedword(X_train, y_train)
		tic = time.clock()
		print('Finished extracting unique words')
		
		newX_train = dfWordAsFeature(X_train, selectedword)
		newX_test = dfWordAsFeature(X_test, selectedword)
		
		lr_test_accuracy = LogResmodel(newX_train, newX_test, y_train, y_test)
		#xgb_test_accuracy = xgboostmodel(newX_train, newX_test, y_train, y_test)
		xgb_test_accuracy = "Coming Soon"
		
		print('Finished training model')
	return render_template("modelresult.html", lr_test = lr_test_accuracy, xgb_test = xgb_test_accuracy, flow_image = full_path)

@app.route('/datauploads', methods=["GET", "POST"])
def datauploads():
	full_path = os.path.join(app.config['UPLOAD_FOLDER'], '1-half.png')
	if request.method == 'POST':
		#for f in request.files.getlist('zipfiles'):
		#	print(f.filename)
		#	f.save(os.path.join('static/uploadstorage', f.filename))
		#print(zipfile)
	
		zipfile = request.files['zipfiles']
		
		zipfilename = secure_filename(zipfile.filename)
		uploadfiles.append(zipfilename)
		zipfile.save(os.path.join('static/uploadstorage', zipfilename))
	return render_template('home.html', uploadfiles = uploadfiles, flow_image = full_path)

@app.route("/inputtest", methods=["GET", "POST"])
def inputtest():
	full_path = os.path.join(app.config['UPLOAD_FOLDER'], '3-half.png')
	return render_template('inputtest.html', title='Testing PDF Doc', flow_image = full_path)

@app.route("/pdftesting", methods=["GET", "POST"])
def pdftesting():
	full_path = os.path.join(app.config['UPLOAD_FOLDER'], '6-half.png')
	if request.method == 'POST' and 'pdffile' in request.files:
		file = request.files['pdffile']
		filename = secure_filename(file.filename)
		file.save(os.path.join('static/uploadstorage', filename))
		
		lr_pred = testingpdf(os.path.join('static/uploadstorage', filename))
	return render_template('pdftesting.html', lrpred = lr_pred, flow_image = full_path)

@app.route("/thankyou", methods=["GET", "POST"])
def thankyou():
	return render_template('thankyou.html', title='Thank You')
if __name__ == '__main__':
	app.run(debug=True)
