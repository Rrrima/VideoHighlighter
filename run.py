import os
from Video import Video
import pandas as pd 
from utils import *
from Highlighter import Highlighter
import random
from timeit import default_timer as timer

if __name__ == '__main__':
	start = timer()
	# vnames = os.listdir('feature_maps')
	ign_list = ['.DS_Store','Air_Force_One','Bearpark_climbing','Cooking','Cockpit_Landing']
	for each in ['Kids_playing_in_leaves']:
		if each in ign_list:
			continue
		print("="*30,each,'='*30)
		vid = Video(each)
		vid.construct_frames()
		frame_feature_list = []
		for f in vid.frames:
			frame_feature_list.append(f.features)
		df = pd.DataFrame(frame_feature_list,columns=vid.fnames)
		df.to_csv(each+'_frame.csv')
		end = timer()
		print(end - start)
		vid.get_superframes()
		super_feature_list = []
		for s in vid.superframes:
			super_feature_list.append(s.features)
		df = pd.DataFrame(super_feature_list,columns=s.super_fnames)
		df.to_csv(each+'_superframe.csv')
		end = timer()
		print(end - start)

	# allvid = all_vid()
	# # choose testing set
	# import random
	# random.seed(0)
	# test_sf = random.choices(allvid,k=3)
	# train_sf = []
	# for v in allvid:
	#     if v not in test_sf:
	#         train_sf.append(v)
	# test_xn,test_yn = prepare_normedsff(test_sf)
	# train_xn,train_yn = prepare_normedsff(train_sf)

	# test super frame model
	# model_super = SuperFrameModel('rf')
	# model_super.eval_model()
	# label,pred = model_super.predict('Cooking')
	# print(pd.DataFrame([label,pred]))

	# test frame model
	# model_frame = FrameModel('rf')
	# model_frame.eval_model()
	# label,pred = model_frame.predict('Cooking')
	# print(pd.DataFrame([label,pred]))

	hl = Highlighter()
	# allvid = all_vid()
	# # test_vid = random.choices(allvid,k=3)
	# highlight_dict = {}
	# xlist = [0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.5,2,4]
	# for x in xlist:
	# 	print("="*20,x,"="*20)
	# 	tempv = []
	# 	for v in allvid:
	# 		print("          test: ",v)
	# 		superlist,vid = hl.predict_video(v)
	# 		peaklist = hl.highlight_video(x)
	# 		tempv.append(len(peaklist))
	# 	highlight_dict[x] = tempv
	# print(highlight_dict)
	# df = pd.DataFrame.from_dict(highlight_dict)
	# df.to_csv('verbosity.csv')
	
	superlist,vid = hl.predict_video('Kids_playing_in_leaves')
	peaklist = hl.highlight_video(1)

	end = timer()
	print(end - start)






