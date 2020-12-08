from Video import Video
from Model import SuperFrameModel,FrameModel
from scipy.signal import find_peaks
import random
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt 

class Highlighter(object):
	def __init__(self):
		print("construct super frame model")
		self.smodel = SuperFrameModel('rf')
		self.smodel.fit_model()
		print("construct single frame model")
		self.fmodel = FrameModel('rf')
		self.fmodel.fit_model()
		self.video = None
		self.super_label = None
		self.super_score = None
		self.frame_label = None
		self.frame_score = None
		self.verbosity = None
		self.peaklist = None

	def predict_video(self,vname):
		self.video = Video(vname)
		self.verbosity = self.video.N/20
		self.super_label,self.super_score = self.smodel.predict(vname)
		self.frame_label, self.frame_score = self.fmodel.predict(vname)
		cutlist = self.video.cutlist
		superlist = []
		for i in range(len(cutlist)-1):
			start = cutlist[i]
			end = cutlist[i+1]
			s = {}
			s['start'] = start
			s['end'] = end
			s['score'] = self.super_score[i]
			s['img'] = self.video.frame_imgs[i]
			superlist.append(s)
		return superlist,self.video

	def highlight_video(self,v=1):
		peaklist = []
		self.verbosity = self.video.N/20
		self.verbosity = self.verbosity/v
		peaks, _ = find_peaks(self.frame_score,distance=self.verbosity)
		for each in peaks:
			p = {}
			p['frame'] = each
			p['score'] = self.frame_score[each]
			p['img'] = self.video.frame_imgs[each]
			peaklist.append(p)
		self.peaklist = peaklist
		return peaklist

	def compare_prediction(self):
		if self.video:
			plt.figure(figsize=(20,2))
			plt.plot(range(self.video.N),self.video.gt)
			plt.plot(range(self.video.N),self.frame_score)
			for c in self.video.cutlist:
			    plt.axvline(x = c, color ='k',alpha=0.5,linestyle='--') 
			for p in self.peaklist:
			    plt.axvline(x = p['frame'], color ='r',alpha=p['score']) 
			plt.title(self.video.name) 
			plt.show()











