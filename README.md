#Preparing NER Model
1. https://nlp.stanford.edu/software/stanford-ner-2018-10-16.zip
2. Unzip the models on the same folder with python des

#Prepare Dataset to Train
1. Prepare a set of PDF files
2. Separate PDF files based on its topic (eg. sport, law) and put it in a folder
3. Zip all folders from step 2
4. You can check the prepared dataset in folder static/dummyuploadfolder
5. If you want to scrape the full dataset from here static/dummyuploadfolder. You can run "scrape.py"

#RUN
1. start flaskblog.py
2. copy the address and open it on browser
3. Upload zip files to newly opened web-page
4. Start training the model
5. Test the model by uploading a PDF files to the next opened web-page