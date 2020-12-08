from sklearn.model_selection import train_test_split
from sklearn.base import clone 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from utils import *
from Video import Video
import random
import numpy as np

random.seed(1)

class SuperFrameModel(object):
	def __init__(self,modelname,split=False):
		allvid = all_vid()
		self.split = split
		# devide the testing / training videos
		self.modelname = modelname
		if not split:
			self.test_vnames = random.choices(allvid,k=3)
			self.train_vnames = allvid
		else:
			self.test_vnames = random.choices(allvid,k=3)
			train_sf = []
			for v in allvid:
			    if v not in self.test_vnames:
			        train_sf.append(v)
			self.train_vnames = train_sf

		self.trainX,self.trainy = prepare_normedsff(self.train_vnames)
		self.testX,self.testy = prepare_normedsff(self.test_vnames)
		self.modeldict = {
			'rf':RandomForestRegressor(random_state=0,n_estimators=100),
			'mlp':MLPRegressor(random_state=0, max_iter=1000,activation='relu',hidden_layer_sizes=(200)),
			'svr':SVR(kernel='rbf',C=10)
		}
		self.model = self.modeldict[self.modelname]

	def eval_model(self):
	    val_result = -cross_val_score(self.model,self.trainX,self.trainy,cv=3,scoring='neg_mean_absolute_error')
	    print(val_result)
	    return val_result

	def fit_model(self):
		X_train, X_test, y_train, y_test = train_test_split(self.trainX,self.trainy, test_size=0.5, random_state=0)
		self.model.fit(X_train,y_train)

	def predict(self,num):
		if self.split:
			vname = self.test_vnames[num]
		else:
			vname = num
		# print("="*20," get superframe score for "+vname+" "+ "="*20)
		X,label = prepare_normedsff([vname])
		pred = self.model.predict(X)
		return label, pred


class FrameModel(object):
	def __init__(self,modelname,split=False):
		allvid = all_vid()
		self.split = split
		# devide the testing / training videos
		self.modelname = modelname
		if not split:
			self.test_vnames = random.choices(allvid,k=3)
			self.train_vnames = allvid
		else:
			self.test_vnames = random.choices(allvid,k=3)
			train_sf = []
			for v in allvid:
			    if v not in self.test_vnames:
			        train_sf.append(v)
			self.train_vnames = train_sf
		self.trainX,self.trainy = prepare_normedff(self.train_vnames)
		self.testX,self.testy = prepare_normedff(self.test_vnames)
		self.modeldict = {
			'rf':RandomForestRegressor(random_state=0,n_estimators=50),
			'mlp':MLPRegressor(random_state=0, max_iter=1000,activation='relu',hidden_layer_sizes=(100,100)),
			'svr':SVR(kernel='rbf',C=0.01)
		}
		self.model = self.modeldict[self.modelname]

	def eval_model(self):
	    val_result = -cross_val_score(self.model,self.trainX,self.trainy,cv=3,scoring='neg_mean_absolute_error')
	    print(val_result)
	    return val_result

	def fit_model(self):
		X_train, X_test, y_train, y_test = train_test_split(self.trainX,self.trainy, test_size=0.5, random_state=0)
		self.model.fit(X_train,y_train)

	def predict(self,name):
		# self.fit_model()
		if self.split:
			vname = self.test_vnames[name]
		else:
			vname = name
		print("="*20," get frame score for "+vname+" "+ "="*20)
		# vid = Video(vname,20)
		X,label = prepare_normedff([vname])
		pred = self.model.predict(X)
		# plt.figure(figsize=(20,2))
		# plt.plot(range(vid.N),vid.gt)
		# plt.plot(range(vid.N),pred)
		# for c in vid.cutlist:
		#     plt.axvline(x = c, color = 'r',alpha=0.5) 
		# plt.show()
		return label, pred








