from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 


# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
Bootstrap(app)


@app.route("/")
def index():
	return render_template("index.html")

@app.route("/classifier")
def home():
	return render_template("classifier.html")


@app.route('/predict', methods=['POST'])
def predict():
		#from sklearn.feature_extraction.text import TfidfVectorizer# Load our data
	df = pd.read_csv('data/names_dataset.csv')

	df_names = df
	# Replacing All F and M with 0 and 1 respectively
	df_names.sex.replace({'F':0,'M':1},inplace=True)

	Xfeatures =df_names['name']

	# Feature Extraction 
	cv = CountVectorizer()
	X = cv.fit_transform(Xfeatures)
	# Labels
	y = df_names.sex



	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	# Naive Bayes Classifier

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	if request.method == 'POST':
		namequery = request.form['namequery']
		data = [namequery]
		vect = cv.transform(data).toarray()
		my_prediction =-1
		my_prediction = clf.predict(vect)
	return render_template("results.html",prediction = my_prediction,name = namequery.upper())


if __name__ == '__main__':
	app.run(debug=True)


