from itertools import chain

from tika import parser
import os
from os.path import isfile, join
import csv

import pandas as pd
from collections import OrderedDict
import unicodedata
import numpy as np 

import re
import zipfile

import errno, os, stat, shutil
#https://stackoverflow.com/questions/1213706/what-user-do-python-scripts-run-as-in-windows
def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

def removing_name_entities(rawtext, ner):
	row = []
	for word in word_tokenize(rawtext):
		fltrword = filterword(word, ner)
		if fltrword != None:
			row.append(fltrword)
	return(row)

def readpdf(path):
	file_data = parser.from_file(path)
	text = file_data['content'].strip()
	text = re.sub('\n|\n1|\t', ' ', text)
	text = re.sub(' +', ' ', text)
	return(text)
	
def extracttext(mypath):
	filename = []
	data = []
	label = []
	for fzip in os.listdir(mypath):
		#Get ZipFile path
		zippath = mypath + '\\' + fzip
		#Get Zipfile name
		zipname = re.sub('\.zip', '', fzip)
		#folder path for extracted zip
		extzippath = mypath + '\\' + zipname
		#Create folder for that processed topic
		os.mkdir(extzippath)
		#Extracting zip
		with zipfile.ZipFile(zippath, 'r') as zip_ref:
			zip_ref.extractall(extzippath)
		for f in os.listdir(extzippath):
			if isfile(join(extzippath, f)):
				path = extzippath + '\\' + f
				text = readpdf(path)
				data.append(text)
				filename.append(f)
				label.append(zipname)
		os.remove(zippath)
		shutil.rmtree(extzippath, ignore_errors=False, onerror=handleRemoveReadonly)

	return(filename, data, label)