import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import pandas as pd
from utils import *

# a class for feature maps for each frame
class Frame(object):
	def __init__(self,i,n,cutlist,img,salmap,gt):
		self.p = i/n * 100
		self.pc = get_pc(i,cutlist)
		self.img = img
		self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		self.salmap = salmap
		self.gt = gt
		self.sift_kp = self.get_sift()
		self.kp_map,self.angle_map,self.size_map = self.get_siftkp_map()
		self.color_hist = self.get_hist()
		self.features = self.construct_features()
		# plt.imshow(self.kp_map)
		# plt.show()

	def get_sift(self):
		sift = cv2.SIFT_create()
		kp = sift.detect(self.gray,None)
		return kp

	def get_siftkp_map(self):
		kp_map = np.zeros_like(self.gray)
		angle_map = np.zeros_like(self.gray)
		size_map = np.zeros_like(self.gray)
		for kp in self.sift_kp:
			pt = kp.pt
			kp_map[int(pt[1])][int(pt[0])] = 1
			angle_map[int(pt[1])][int(pt[0])] = kp.angle
			size_map[int(pt[1])][int(pt[0])] = kp.size
		return kp_map,angle_map,size_map

	def get_hist(self):
		hist_dict = {}
		gray_hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
		gray_hist = np.squeeze(gray_hist, axis=1)
		hist_dict['gray'] = gray_hist
		chans = cv2.split(self.img)
		colors = ("b", "g", "r")
		for (chan, color) in zip(chans, colors):
			hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
			hist = np.squeeze(hist, axis=1)
			hist_dict[color] = hist
		return hist_dict

	def construct_features(self):
		kps = [len(self.sift_kp),np.mean(self.angle_map),np.mean(self.size_map)]
		chans = cv2.split(self.img)
		bgrg = [np.mean(chans[0]),np.mean(chans[1]),np.mean(chans[2]),np.mean(self.gray)]
		f = [self.gt,self.p,self.pc,np.mean(self.salmap)]
		f.extend(kps)
		f.extend(bgrg)
		return f

class SuperFrame(object):
	def __init__(self,frames,start,end,N,fnames):
		self.frames = frames
		self.start = start/N
		self.end = end/N
		self.duration = self.end - self.start
		self.fnames = fnames
		self.frame_features = self.get_frame_features()
		self.features = self.get_super_features()
		self.super_fnames = self.get_fnames()

	def get_frame_features(self):
		flist = []
		for f in self.frames:
			flist.append(f.features)
		return pd.DataFrame(flist,columns=self.fnames)

	def get_super_features(self):
		score = self.frame_features.mean()[0]
		means = self.frame_features.mean()[3:]
		stds = self.frame_features.std()[3:]
		base = [score,self.start,self.end,self.duration]
		base.extend(means)
		base.extend(stds)
		return base

	def get_fnames(self):
		means = ['mean_'+x for x in list(self.frame_features.mean()[3:].index)]
		stds = ['std_'+x for x in list(self.frame_features.std()[3:].index)]
		base = ['score','start','end','duration']
		base.extend(means)
		base.extend(stds)
		return base

















