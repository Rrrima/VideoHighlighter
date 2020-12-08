import os
import cv2
import torch
from torch.utils import data
from torchvision.transforms import functional as F
import numpy as np

class ImageDataTest(data.Dataset):
    def __init__(self, image_frames):
        self.image_frames = image_frames
        self.image_num = len(self.image_frames)

    def __getitem__(self, item):
        image, im_size = load_image_test(image_frames[item])
        image = torch.Tensor(image)
        return {'image': image,'size': im_size}

    def __len__(self):
        return self.image_num

def get_loader(image_frames):
    dataset = ImageDataTest(image_frames)
    data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    return data_loader

def load_image_test(im):
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size
