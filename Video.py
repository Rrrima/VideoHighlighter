import os
import cv2
from scipy.io import loadmat
from statistics import mean
import numpy as np
from Frame import Frame,SuperFrame
from utils import *

# class for each video
class Video(object):
	def __init__(self,videoname,thred=20):
		self.thred = thred
		self.name = videoname
		print("loading frames ...")
		self.frame_imgs = self.frame_capture()
		self.N = len(self.frame_imgs)-1
		print("loading salient maps ...")
		self.salmaps = self.load_salmaps()
		print("get ground truth score ...")
		self.gt = self.load_gt()
		print("split super frames ...")
		self.cutlist = self.get_cutlist()
		self.fnames = fnames()
		self.frames = None
		self.superframes = None

	def frame_capture(self):
		path = os.path.join('videos',self.name+'.mp4')
		vidObj = cv2.VideoCapture(path) 
		success = 1
		imgarray = []
		count = 0            
		while success: 
			success, img = vidObj.read() 
			# get the images for each of the 10 frame
			if count%10 == 0:
				imgarray.append(img)
			count += 1
		return imgarray

	def load_salmaps(self):
		path = os.path.join('feature_maps',self.name)
		names = ['frame{}.{}'.format(i*10,'png') for i in range(0,self.N)]
		pathnames = [os.path.join(path,name) for name in names]
		imgs = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in pathnames]
		return imgs

	def load_gt(self):
		x = loadmat('GT/'+self.name+'.mat')
		gt = np.squeeze(x['gt_score'], axis=1)
		gt_list = []
		for i in range(0,self.N):
			start = i*10;
			end = (i+1)*10
			gt_list.append(mean(gt[start:end+1]))
		return gt_list

	def construct_frames(self):
		print("  constructing frame ...")
		framelist = []
		k = 0
		for i in range(self.N):
			print("      === frame "+str(i)+"/"+str(self.N)+" ===")
			cur_frame = Frame(i,self.N,self.cutlist,self.frame_imgs[i],
				self.salmaps[i],self.gt[i])
			framelist.append(cur_frame)
		self.frames = framelist

	def get_cutlist(self):
		salmaps = self.salmaps
		salmaps_resized = resizeimgs(salmaps)
		cutlist = [0]
		corlist = []
		for i in range(1,len(salmaps_resized)):
		    pref = salmaps_resized[i-1]
		    curf = salmaps_resized[i]
		    if curf.any()!= 0:
		        corval = np.inner(pref,curf).sum()
		        corlist.append(corval)
		thred = np.percentile(corlist,self.thred)
		for i in range(1,len(salmaps_resized)):
		    pref = salmaps_resized[i-1]
		    curf = salmaps_resized[i]
		    if curf.any()!= 0:
		        corval = np.inner(pref,curf).sum()
		        corlist.append(corval)
		        if corval<thred and (i-cutlist[-1]>self.N/16):
		            print((i,corval))
		            cutlist.append(i)
		cutlist.append(self.N)
		return cutlist

	def get_superframes(self):
		print("  constructing super frame ...")
		sfs = []
		cutlist = self.cutlist
		if not self.frames:
			self.construct_frames()
		for i in range(len(cutlist)-1):
			print("      === frame "+str(i)+"/"+str(len(cutlist)-1)+" ===")
			frames = self.frames[cutlist[i]:cutlist[i+1]]
			sf = SuperFrame(frames,cutlist[i],cutlist[i+1],self.N,self.fnames)
			sfs.append(sf)
		self.superframes = sfs


if __name__ == '__main__':
	vid = Video('Cooking')
	print(vid.fnames)

