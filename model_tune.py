import os
import random
from Video import Video
import pandas as pd 
from utils import *
from Model import SuperFrameModel,FrameModel
from sklearn.metrics import mean_absolute_error

allvid = all_vid()
test_vid = random.choices(allvid,k=10)

total_list = []
model_super = FrameModel('rf')
model_super.fit_model()
mae_list = []
for vname in test_vid:
    label,pred = model_super.predict(vname)
    mae_list.append(mean_absolute_error(label,pred))
print(mae_list)
total_list.append(mae_list)
print(np.mean(mae_list))
print(np.mean(model_super.eval_model()))

model_super = FrameModel('svr')
model_super.fit_model()
mae_list = []
for vname in test_vid:
    label,pred = model_super.predict(vname)
    mae_list.append(mean_absolute_error(label,pred))
print(mae_list)
total_list.append(mae_list)
print(np.mean(mae_list))

model_super = FrameModel('mlp')
model_super.fit_model()
mae_list = []
for vname in test_vid:
    label,pred = model_super.predict(vname)
    mae_list.append(mean_absolute_error(label,pred))
print(mae_list)
total_list.append(mae_list)
print(np.mean(mae_list))

total_list = np.transpose(np.array(total_list))

df = pd.DataFrame(total_list,columns=['rf','svr','mlp'])
df.to_csv('singleframe_comparison.csv')

print(total_list)

