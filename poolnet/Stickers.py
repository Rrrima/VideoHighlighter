import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class Stickers(object):
	def __init__(self,name,salmaps,images):
		self.name = name
		# list of salient maps
		self.salmaps = salmaps
		# list of original images
		self.images = images

	def get_sticker(self,fstart):
		frame = Frame(self.salmaps[fstart],self.images[fstart])
		frame.get_sticker[0]

	def get_caption(self,fstart):
		pass


class Frame(object):
	def __init__(self,salmap,image,comprate=0.1,smoothing=0):
		self.salmap = salmap
		self.image = image
		self.comprate = comprate
		self.smoothing = smoothing
		_, self.mask = self.get_pointlist(self.salmap)
		self.compsal = compress()
		self.object_groups = get_object()

	def compress(self):
		rate = self.comprate
		width = int(self.salmap.shape[1] * rate)
		height = int(self.salmap.shape[0] * rate)
		dim = (width, height)
		resized = cv2.resize(self.salmap, dim, interpolation = cv2.INTER_AREA)
		return resized

	def get_pointlist(self,im):
		im_trans = im.copy()
		point_list = []
		for i in range(im.shape[1]):
			for j in range(im.shape[0]):
				if im[j][i] <40:
					im_trans[j][i] = 0
				else:
					im_trans[j][i] = 1
					point_list.append([j,i])
		return point_list,im_trans

	def get_grouping_dict(self,pts,labels):
		pts = np.array(pts)
		groups = defaultdict(list)
		all_labels = set(labels)
		for i in range(len(labels)):
			if labels[i] != -1:
				groups[labels[i]].append(pts[i])
		return groups 

	def get_object(self):
		if not self.smoothing:
			_,mask = self.get_pointlist(self.salmap)
			point_list_all,_ = self.get_pointlist(self.compsal)
			clustering = DBSCAN(eps=3, min_samples=10,metric='euclidean').fit(point_list_all)
			groups = self.get_grouping_dict(point_list_all,clustering.labels_)
			return groups

	def get_boundingbox(self):
		bounds = {}
		rate = self.comprate
		for k,v in groups.items():
			pty = [vv[0] for vv in v]
			ptx = [vv[1] for vv in v]
			# print(len(pty))
			xmin = np.min(ptx) * rate
			xmax = np.max(ptx) * rate
			ymin = np.min(pty) * rate
			ymax = np.max(pty) * rate
			bounds[k] = [xmin,ymin,xmax,ymax]
		return bounds

	def get_sticker(self,id):
		pts = self.object_groups[id]
		single_mask = np.zeros_like(self.compsal)
		for p in pts:
			single_mask[p[0],p[1]] = 1
		plt.imshow(single_mask)





		










