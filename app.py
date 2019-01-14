from flask import Flask,render_template,request
	
	
	
import numpy as np
import pandas as pd
import csv
import os
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('frntproject1.html')

	

@app.route('/page',methods=['GET','POST'])
def page():
	data = {}
	if request.form:

		form_data = request.form
		data['form'] = form_data
		year = form_data['year']
		month = form_data['month']
		day = form_data['day']
		datobj = datetime.datetime.strptime(form_data['date'], '%Y-%m-%d')
		date = datetime.datetime.strftime(datobj,'%Y%m%d')
		carrier = form_data['carrier']
		origin = form_data['origin']
		dest = form_data['dest']
		print(date)
		
		#datee=datetime.date(2002, 12,4)
		#datee.strftime('%m%d%Y')
		#date = datetime.date(2002, 12,4).strftime("%Y%m%d")
		print(data)

		
		fdata = pd.read_csv('564220792_T_ONTIME.csv')
		fdata1=pd.DataFrame(columns=['YEAR','MONTH','DAY_OF_MONTH','FL_DATE','CARRIER','ORIGIN','DEST','ARR_DEL15'])
		fdata1['YEAR']=fdata['YEAR']
		fdata1['MONTH']=fdata['MONTH']
		fdata1['DAY_OF_MONTH']=fdata['DAY_OF_MONTH']
		fdata1['FL_DATE']=fdata['FL_DATE']
		fdata1['CARRIER']=fdata['CARRIER']
		fdata1['ORIGIN']=fdata['ORIGIN']
		fdata1['DEST']=fdata['DEST']
		fdata1['ARR_DEL15']=fdata['ARR_DEL15']
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		fdata1["Carrier_Name"] = le.fit_transform(fdata1["CARRIER"])
		Carrier = list(le.classes_)
		fdata1["Origin_Point"] = le.fit_transform(fdata1["ORIGIN"])
		Origin = list(le.classes_)
		fdata1["Destination"] = le.fit_transform(fdata1["DEST"])
		Dest = list(le.classes_)
		fdata1.drop(['CARRIER','ORIGIN','DEST'], axis=1, inplace=True)#Removing original encoded columns
		fdata1["FL_DATE"] = fdata1["FL_DATE"].apply(lambda x: int(''.join(x.split("-"))))
		np.random.seed(10)
		Delay_YesNo = fdata1['ARR_DEL15']
		fdata1.drop(['ARR_DEL15'], axis=1, inplace=True)#Removing target variable
		data_part2 = pd.DataFrame(fdata1)
		Delay_YesNo1=np.nan_to_num(Delay_YesNo)
		from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
		from sklearn.ensemble import RandomForestClassifier
		from sklearn import cross_validation
		from sklearn.metrics import confusion_matrix, roc_curve
		import matplotlib
		from matplotlib import pyplot as plt
		import matplotlib.pyplot as plt
		X_train, X_test, Y_train, Y_test = train_test_split(data_part2, Delay_YesNo1, test_size=0.2, random_state=42)
		#startTimeGS = datetime.now()
		from sklearn.grid_search import GridSearchCV
		rf = RandomForestClassifier()
		rf.fit(X_train, Y_train)
		#startTimeRF = datetime.now()
		#cv = cross_validation.KFold(len(X_train), n_folds=3, shuffle=True, random_state=2)
		#cvScores = cross_val_score(rf, X_train, Y_train, cv=cv)
		#rf.fit(X_train, Y_train)
		df=pd.read_csv("564220793_T_ONTIME.csv")
		fdata3=pd.DataFrame(columns=['YEAR','MONTH','DAY_OF_MONTH','FL_DATE','CARRIER','ORIGIN','DEST'])
		fdata3['YEAR']=df['YEAR']
		fdata3['MONTH']=df['MONTH']
		fdata3['DAY_OF_MONTH']=df['DAY_OF_MONTH']
		fdata3['FL_DATE']=df['FL_DATE']
		fdata3['CARRIER']=df['CARRIER']
		fdata3['ORIGIN']=df['ORIGIN']
		fdata3['DEST']=df['DEST']
		fdata3["FL_DATE"] = fdata3["FL_DATE"].apply(lambda x: int(''.join(x.split("-"))))#Formatting date for convinience
		from sklearn.preprocessing import LabelEncoder
		se = LabelEncoder()
		fdata3["Carrier_Name"] = se.fit_transform(fdata3["CARRIER"])
		Carrier = list(se.classes_)
		fdata3["Origin_Point"] = se.fit_transform(fdata3["ORIGIN"])
		Origin = list(se.classes_)
		fdata3["Destination"] = se.fit_transform(fdata3["DEST"])
		Dest = list(se.classes_)
		fdata3.drop(['CARRIER','ORIGIN','DEST'], axis=1, inplace=True)
		input_data = np.array([year,month,day,date,carrier,origin,dest])
		print(input_data)
		fdata4=rf.predict(input_data.reshape(1,-1))
		print(fdata4)
		return render_template("test.html",prediction=fdata4)


if __name__ == '__main__':
	app.run(debug = True)