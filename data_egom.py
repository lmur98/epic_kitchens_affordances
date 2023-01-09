import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import PIL.Image as pil
import torch
import torch.utils.data as data
from torchvision import transforms
import glob
import collections
import cv2
import open3d as o3d
import pandas as pd
from read_write_model import read_model
from utils_read_annotations import EP100_and_VISOR_annotations


Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])

class VideoSequentialDataset(data.Dataset):
    """Superclass for sequential images dataloaders
    """
    def __init__(self, data_path, kitchen, height, width, frame_idxs):
        super(VideoSequentialDataset, self).__init__()
        self.colmap_poses = os.path.join(data_path, kitchen,'colmap')
        self.masks = os.path.join(data_path, kitchen, 'selected_plus_guided_masks')
        self.rgb = os.path.join(data_path, kitchen, 'selected_plus_guided_rgb')
        
        self.filenames = self.read_directory()
        self.height = height
        self.width = width
        self.colors = self.get_colormap()
        self.VISOR_path = '/home/lmur/Desktop/VISORS_Kitchen_Dataset'
        self.EP100_and_VISOR_reader = EP100_and_VISOR_annotations(self.VISOR_path, self.rgb, kitchen)
        self.frame_idxs = frame_idxs
        
        self.cameras_Colmap, self.imgs_Colmap, self.pts_Colmap = read_model(self.colmap_poses, ext=".txt")
        self.fx = self.cameras_Colmap[1].params[0]
        self.fy = self.cameras_Colmap[1].params[1]
        self.cx = self.cameras_Colmap[1].params[2]
        self.cy = self.cameras_Colmap[1].params[3]
        

    def read_directory(self):
        paths = glob.glob(os.path.join(self.rgb, '*.jpg'))
        paths.sort()
        return paths

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        full_filename = self.filenames[index]
        for i in self.frame_idxs:
            inputs[("color", i)] = self.get_color(self.filenames[index + i])
        inputs["full_filename"] = full_filename
        inputs["filename"] = full_filename.split('/')[-1] 
        print(full_filename.split('/')[-1].split('_')[0:2])
        sequence = full_filename.split('/')[-1].split('_')[0:2]
        #Join the two string elements of the list with a '_' in the middle
        inputs['sequence'] = '_'.join(sequence)
        inputs["subset"] = 'train'
        inputs["aff_annotation"], inputs["EP100_annotation"], inputs['VISOR_annotation'] = self.EP100_and_VISOR_reader.affordance_hotspot(inputs["filename"], inputs['subset'], inputs['sequence'])
        inputs["exists_affordance"] = self.check_exits_affordance(inputs["aff_annotation"])
        return inputs

    def check_exits_affordance(self, aff_annotation):
        if aff_annotation is not None: #We have an annotation on EP100
            if len(aff_annotation['interacting_objects']) > 0: #The IoU is above the threshold
                return True
        return False

    def get_color(self, filename):
        img = cv2.imread(filename)
        return img

    def get_mask(self, filename):
        mask = cv2.imread(filename.replace('sampled_rgb', 'sampled_masks').replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
        return mask


    def get_colormap(self, N=256, normalized = False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap
        cmap_dict = {}
        for i in range(N):
            cmap_dict[i] = [cmap[i,0], cmap[i,1], cmap[i, 2]]
        return cmap_dict

