# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:33:11 2021
"The goal of this script is to divide the 3D images into smaller patches to fit the gpu memory of the computer"
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


root_directory = "Data/"

patch_size = 256


#%% For images 
img_dir = root_directory+"BraTS2020_TrainingData/input_data_128"

img_dir_train = root_directory+"BraTS2020_TrainingData/input_data_128/train/images"
img_dir_val = root_directory+"BraTS2020_TrainingData/input_data_128/val/images"
img_dir_test = root_directory+"BraTS2020_TrainingData/input_data_128/test/images"


img_dir_train = root_directory+"BraTS2020_TrainingData/input_data_128/train/images"
images = os.listdir(img_dir_train)
for i, image_name in enumerate(images):  
    if (image_name.endswith(".npy") and (image_name[0]=='i') ) :
        #print(path+"\\"+image_name)
        image = np.load(img_dir_train+"\\"+image_name)  
        #print(image.shape)
        print("Now patchifying image:", img_dir_train+"\\"+image_name)
        for k in range(image.shape[-1]):
            patches_img = patchify(image[:,:,:,k], (64, 64, 64), step=64)  
            print(patches_img.shape)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    for o in range(patches_img.shape[2]):
                        
                        single_patch_img = patches_img[i,j,o,:,:,:]
                        single_patch_img = (single_patch_img.astype('float32')) 
                                                      
                        
                        np.save(root_directory+"BraTS2020_TrainingData/input_data_128/train/64_patches\images\\"+image_name[:-4]+"_"+str(k)+"patch_"+str(i)+str(j)+str(o)+".npy", single_patch_img)
                        
                    
img_dir_val = root_directory+"BraTS2020_TrainingData/input_data_128/val/images"
images = os.listdir(img_dir_val)
for i, image_name in enumerate(images):  
    if (image_name.endswith(".npy") and (image_name[0]=='i') ) :
       
        image = np.load(img_dir_val+"\\"+image_name)  
        
        for k in range(image.shape[-1]):
            patches_img = patchify(image[:,:,:,k], (64, 64, 64), step=64)  
            
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    for o in range(patches_img.shape[2]):
                        
                        single_patch_img = patches_img[i,j,o,:,:,:]
                        single_patch_img = (single_patch_img.astype('float32')) 
                                                      
                        
                        np.save(root_directory+"BraTS2020_TrainingData/input_data_128/val/64_patches\images\\"+image_name[:-4]+"_"+str(k)+"patch_"+str(i)+str(j)+str(o)+".npy", single_patch_img)
          
img_dir_test = root_directory+"BraTS2020_TrainingData/input_data_128/test/images"
images = os.listdir(img_dir_test)
for i, image_name in enumerate(images):  
    if (image_name.endswith(".npy") and (image_name[0]=='i') ) :
        image = np.load(img_dir_test+"\\"+image_name)  

        for k in range(image.shape[-1]):
            patches_img = patchify(image[:,:,:,k], (64, 64, 64), step=64)  
            
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    for o in range(patches_img.shape[2]):
                        
                        single_patch_img = patches_img[i,j,o,:,:,:]
                        single_patch_img = (single_patch_img.astype('float32')) 
                                        
                        
                        np.save(root_directory+"BraTS2020_TrainingData/input_data_128/test/64_patches\images\\"+image_name[:-4]+"_"+str(k)+"patch_"+str(i)+str(j)+str(o)+".npy", single_patch_img)
                              
  

# %% Testing
img_dir_val1  = root_directory+"BraTS2020_TrainingData/input_data_128/val/64_patches\images\\"    
L = os.listdir(img_dir_val1)
img_test = np.load(img_dir_val1+L[0])

print (img_test.shape)
train_path = root_directory+"BraTS2020_TrainingData/input_data_128/train/64_patches/images//"
L=os.listdir(train_path)

print (L[3][-13:])

# %% Regrouping patches

train_path = root_directory+"BraTS2020_TrainingData/input_data_128/train/64_patches/images//"
L=os.listdir(train_path)
for img in range(len(L)-16):  
    if( ( L[img][:-15] == L[img+8][:-15] ) and (L[img+8][:-15] == L[img+16][:-15] ) ):
        test_image_flair = np.load(train_path+L[img])
        test_image_t1ce = np.load(train_path+L[img+8])
        test_image_t2 = np.load(train_path+L[img+16])
        
        combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)
        
        np.save(root_directory+"BraTS2020_TrainingData/Patched_data/train/images\\"+L[img][:-15]+"_"+L[img][-13:], combined_x)

val_path = root_directory+"BraTS2020_TrainingData/input_data_128/val/64_patches/images//"
L=os.listdir(val_path)
for img in range(len(L)-16):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
    if( ( L[img][:-15] == L[img+8][:-15] ) and (L[img+8][:-15] == L[img+16][:-15] ) ):
        temp_combined_images = np.stack([np.load(val_path+L[img]), np.load(val_path+L[img+8]), np.load(val_path+L[img+16])], axis=3)
        
        np.save(root_directory+"BraTS2020_TrainingData/Patched_data/val/images\\"+L[img][:-15]+"_"+L[img][-13:], temp_combined_images)

    
test_path = root_directory+"BraTS2020_TrainingData/input_data_128/test/64_patches/images//"
L=os.listdir(test_path)
for img in range(len(L)-16):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)
    if( ( L[img][:-15] == L[img+8][:-15] ) and (L[img+8][:-15] == L[img+16][:-15] ) ):
        temp_combined_images = np.stack([np.load(test_path+L[img]), np.load(test_path+L[img+8]), np.load(test_path+L[img+16])], axis=3)
        
        np.save(root_directory+"BraTS2020_TrainingData/Patched_data/test/images\\"+L[img][:-15]+"_"+L[img][-13:], temp_combined_images)
 
    
# %% checking dimesions
train_path = root_directory+"BraTS2020_TrainingData/Patched_data/test/images\\"
images = os.listdir(train_path)

L=os.listdir(train_path)
img_test = np.load(train_path+L[0])

print (img_test.shape)
             
#Testing compatibility :
    
if(( L[img][:-15] == L[img+8][:-15] ) and (L[img+8][:-15] == L[img+16][:-15] )) :
    print(L[k],L[k+8],L[k+16])    

#%% For masks 

mask_dir_train = root_directory+"BraTS2020_TrainingData/input_data_128/train/masks"
images = os.listdir(mask_dir_train)
for i, mask_name in enumerate(images):  
    if (mask_name.endswith(".npy")  and (mask_name[0]=='m') ) :
        #print(path+"\\"+image_name)
        mask = np.load(mask_dir_train+"\\"+mask_name)  
        mask=np.argmax(mask, axis=3)
        patches_img = patchify(mask, (64, 64, 64), step=64) 
        #print(image.shape)
        #print("Now patchifying image:", img_dir_test+"\\"+image_name)
             
            
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                for o in range(patches_img.shape[2]):
                    
                    single_patch_img = patches_img[i,j,o,:,:,:]
                    single_patch_img = (single_patch_img.astype('float32')) #We will preprocess using one of the backbones
                    #single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                    single_patch_img = to_categorical(single_patch_img, num_classes=4)
                    np.save(root_directory+"BraTS2020_TrainingData/Patched_data/train/masks\\"+mask_name[:-4]+"_"+"patch_"+str(i)+str(j)+str(o)+".npy", single_patch_img)
                    #image_dataset.append(single_patch_img)    

mask_dir_val = root_directory+"BraTS2020_TrainingData/input_data_128/val/masks"
images = os.listdir(mask_dir_val)                    
for i, mask_name in enumerate(images):  
    if (mask_name.endswith(".npy")  and (mask_name[0]=='m') ) :
        #print(path+"\\"+image_name)
        mask = np.load(mask_dir_val+"\\"+mask_name)  
        mask=np.argmax(mask, axis=3)
        patches_img = patchify(mask, (64, 64, 64), step=64) 
        #print(image.shape)
        #print("Now patchifying image:", img_dir_test+"\\"+image_name)
             
            
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                for o in range(patches_img.shape[2]):
                    
                    single_patch_img = patches_img[i,j,o,:,:,:]
                    single_patch_img = (single_patch_img.astype('float32')) #We will preprocess using one of the backbones
                    #single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                    single_patch_img = to_categorical(single_patch_img, num_classes=4)
                    np.save(root_directory+"BraTS2020_TrainingData/Patched_data/val/masks\\"+mask_name[:-4]+"_"+"patch_"+str(i)+str(j)+str(o)+".npy", single_patch_img)
                    #image_dataset.append(single_patch_img)    

                    
mask_dir_test = root_directory+"BraTS2020_TrainingData/input_data_128/test/masks"
images = os.listdir(mask_dir_test)                    
for i, mask_name in enumerate(images):  
    if (mask_name.endswith(".npy")  and (mask_name[0]=='m') ) :
        #print(path+"\\"+image_name)
        mask = np.load(mask_dir_test+"\\"+mask_name)  
        mask=np.argmax(mask, axis=3)
        patches_img = patchify(mask, (64, 64, 64), step=64) 
        #print(image.shape)
        #print("Now patchifying image:", img_dir_test+"\\"+image_name)
             
            
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                for o in range(patches_img.shape[2]):
                    
                    single_patch_img = patches_img[i,j,o,:,:,:]
                    single_patch_img = (single_patch_img.astype('float32')) #We will preprocess using one of the backbones
                    #single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                    single_patch_img = to_categorical(single_patch_img, num_classes=4)
                    np.save(root_directory+"BraTS2020_TrainingData/Patched_data/test/masks\\"+mask_name[:-4]+"_"+"patch_"+str(i)+str(j)+str(o)+".npy", single_patch_img)
                    #image_dataset.append(single_patch_img)    
#%% testing for masks                    
mask_dir_train = root_directory+"BraTS2020_TrainingData/input_data_128/train/masks"
L=os.listdir(mask_dir_train)
img_test = np.load(mask_dir_train+L[0])
print (img_test.shape)                    
mask_dir_train = root_directory+"BraTS2020_TrainingData/Patched_data/train/masks\\"
L=os.listdir(mask_dir_train)
img_test = np.load(mask_dir_train+L[77])
print (img_test.max())

