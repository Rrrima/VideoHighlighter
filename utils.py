import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import pandas as pd
from sklearn import preprocessing
import os

def plotseq(imgs,s,e):
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    fig, ax = plt.subplots(1,7,figsize=(30,15))
    plt.title('frame{} - frame{}'.format(s*10,e*10),fontsize=20)
    ax[0].imshow(imgs[0])
    ax[0].axis('off')
    ax[1].imshow(imgs[1])
    ax[1].axis('off')
    ax[2].imshow(imgs[2])
    ax[2].axis('off')
    ax[3].imshow(imgs[3])
    ax[3].axis('off')
    ax[4].imshow(imgs[4])
    ax[4].axis('off')
    ax[5].imshow(imgs[5])
    ax[5].axis('off')
    ax[6].imshow(imgs[6])
    ax[6].axis('off')
    plt.tight_layout()
    plt.show()

def resizeimgs(imgs,size=(140,90)):
    return [cv2.resize(img, size, interpolation = cv2.INTER_AREA)/255 for img in imgs]


def get_pc(cur,cutlist):
	for i in range(len(cutlist)-1):
		if (cur >= cutlist[i]) and (cur <=cutlist[i+1]):
			return (cur-cutlist[i])/(cutlist[i+1]-cutlist[i])

def fnames():
	basic = ['score','percentile','super_percentile','mean_sal']
	sift = ['keypoints','kp_angle','kp_size']
	color = ['red','green','blue','gray']
	fname = []
	fname.extend(basic)
	fname.extend(sift)
	fname.extend(color)
	return fname

def get_ff(name):
	path = 'features/'+ name + '_frame.csv'
	df = pd.read_csv(path)
	return df.drop(df.columns[0],axis=1) 

def get_sff(name):
	path = 'features/'+ name + '_superframe.csv'
	df = pd.read_csv(path)
	return df.drop(df.columns[0],axis=1)

def col_mmnorm(df):
	normalized_df=(df-df.min())/(df.max()-df.min())
	return normalized_df

def col_xnorm(df):
	normalized_df=(df-df.mean())/df.std()
	return normalized_df

def all_vid():
	names = os.listdir('videos')
	nlist = []
	for each in names:
		if 'mp4' in each:
			nlist.append(each.replace('.mp4',''))
	return nlist

def prepare_sff(names):
	flist = []
	for n in names:
		flist.append(get_sff(n))
	df = pd.concat(flist).reset_index(drop=True)
	y = df['score']
	x = df.drop(['score'],axis=1)
	return x,y

def prepare_normedsff(names):
	flist = []
	llist = []
	for n in names:
		temp = get_sff(n)
		y = np.array(temp['score']) + 0.000001
		x1 = temp[['start','end','duration']]
		x2 = col_mmnorm(temp.drop(['score','start','end','duration'],axis=1))
		x = pd.concat([x1,x2],axis = 1)
		flist.append(x)
		llist.extend(y)
	features = pd.concat(flist).reset_index(drop=True)
	return features,llist


def prepare_normedff(names):
	flist = []
	llist = []
	for n in names:
		temp = get_ff(n)
		y = list(temp['score'])
		x1 = temp[['percentile','super_percentile']]
		x2 = col_xnorm(temp.drop(['score','percentile','super_percentile'],axis=1))
		x = pd.concat([x1,x2],axis = 1)
		flist.append(x)
		llist.extend(y)
	features = pd.concat(flist).reset_index(drop=True)
	return features,llist