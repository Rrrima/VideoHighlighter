import torch
from torch.nn import utils, functional as F
from .networks.joint_poolnet import build_model
import numpy as np
import os
import cv2
import time

class Solver(object):
    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.build_model()
        print('Loading pre-trained model from %s...' % model)
        self.net.load_state_dict(torch.load('poolnet/networks/model/sal_model.pth'))
        self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model()
        self.net = self.net.cuda()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        self.net.base.load_pretrained_model(torch.load('poolnet/networks/model/resnet50_caffe.pth'))
        # self.print_network(self.net, 'PoolNet Structure')

    def test(self):
        EPSILON = 1e-8
        time_s = time.time()
        img_num = len(self.test_loader)
        sal_maps = []
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()
                preds = self.net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                multi_fuse = 255 * pred
                sal_maps.append(multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')
        return sal_maps

