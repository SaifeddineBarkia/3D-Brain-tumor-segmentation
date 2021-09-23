# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:36:30 2021

The goal of this script is to delete images that doesn't contain any useful information for the related mask
@author: Saif
"""

import os
import cv2
import numpy as np
import glob

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image
import tensorflow as tf
from tensorflow import keras
#import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
import random
from tensorflow.keras.utils import to_categorical

def load_img(img_dir, image_name):
    if (image_name.split('.')[1] == 'npy'):
        
        image = np.load(img_dir+image_name)
                  
    return(np.float32(image))



def filtering_data(img_dir, img_list, mask_dir, mask_list, save_dir_img, save_dir_mask):
    for index in range(len(img_list)):
        image = load_img(img_dir, img_list[index])
        mask = load_img(mask_dir, mask_list[index])
        
        mask_max = np.argmax(mask, axis=3)
        val, counts = np.unique(mask_max, return_counts=True)
    
        if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
            print("Save Me")
            np.save(save_dir_img+img_list[index]+'.npy', image)
            np.save(save_dir_mask+mask_list[index]+'.npy', mask)
        else:
            print(img_list[index],"I am useless")   
        

#%% filtering the data that we have:
#Train
root_directory = "Data/"
img_dir = root_directory+"BraTS2020_TrainingData/Patched_data/train/images/"
img_list = os.listdir(img_dir)
mask_dir = root_directory+"BraTS2020_TrainingData/Patched_data/train/masks/"
mask_list = os.listdir(mask_dir)
save_dir_img = root_directory+"BraTS2020_TrainingData/Final_data/train/images/"
save_dir_mask = root_directory+"BraTS2020_TrainingData/Final_data/train/masks/"

    
filtering_data(img_dir, img_list, mask_dir, mask_list, save_dir_img, save_dir_mask)

#val

root_directory = "Data/"
img_dir = root_directory+"BraTS2020_TrainingData/Patched_data/val/images/"
img_list = os.listdir(img_dir)
mask_dir = root_directory+"BraTS2020_TrainingData/Patched_data/val/masks/"
mask_list = os.listdir(mask_dir)
save_dir_img = root_directory+"BraTS2020_TrainingData/Final_data/val/images/"
save_dir_mask = root_directory+"BraTS2020_TrainingData/Final_data/val/masks/"

    
filtering_data(img_dir, img_list, mask_dir, mask_list, save_dir_img, save_dir_mask)

#test
root_directory = "Data/"
img_dir = root_directory+"BraTS2020_TrainingData/Patched_data/test/images/"
img_list = os.listdir(img_dir)
mask_dir = root_directory+"BraTS2020_TrainingData/Patched_data/test/masks/"
mask_list = os.listdir(mask_dir)
save_dir_img = root_directory+"BraTS2020_TrainingData/Final_data/test/images/"
save_dir_mask = root_directory+"BraTS2020_TrainingData/Final_data/test/masks/"
filtering_data(img_dir, img_list, mask_dir, mask_list, save_dir_img, save_dir_mask)
    
#%%  Testing
save_dir_img = root_directory+"BraTS2020_TrainingData/Final_data/test/images/"
mask_list = sorted(os.listdir(save_dir_mask))
save_dir_mask = root_directory+"BraTS2020_TrainingData/Final_data/test/masks/"
img_list = sorted(os.listdir(save_dir_img))

img_num = random.randint(0,len(img_list)-1)
test_img = img_list[img_num]
test_mask = mask_list[img_num]

print(test_img,test_mask)

