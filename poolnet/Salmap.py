from .joint_dataset import get_loader
from .joint_solver import Solver
import os

class Salmap(object):
	def __init__(self,image_frames,test_folder):
		self.image_frames = image_frames
		self.test_folder = test_folder
		print('* start loading data')
		self.test_loader = get_loader(self.image_frames)
		print('* start building solver')
		self.solver = Solver(self.test_loader)
		if not os.path.exists(test_loader):
			os.mkdir(test_folder)
		print('* start extracting salmaps')
		self.salmaps = self.get_salmaps()
		print(self.salmaps.shape)

	def get_salmaps(self):
		salmaps = self.solver.test()
		return salmaps


