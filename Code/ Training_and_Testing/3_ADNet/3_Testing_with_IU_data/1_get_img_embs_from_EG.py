## Read Me ##
# Weights trained with model_siamese_RedCRFENet on ROIs of NIHCC data is named "EG_on_triplet_loss.h5" and saved.
# These weights are loaded. IU_cropped_images images are run with a forward pass and the obtained features (16x512) are saved.

import cv2
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np 
import math

import sys
sys.path.insert(1, '/home/ada/Preethi/XRay_Report_Generation/Code/Normal_vs_Abnormal/siamese')
import model_siamese_lstm
import os

import sys
sys.path.insert(1, '../../Networks') 
import model_siamese_RedCRFENet as network

'''
16 patches per image
Training/val/test split from the text files
Load weights on Siamese and get embeddings (16, 512)
'''

def get_patches(img):
    extracted_patches = extract_patches_2d(img, (128, 128))

    nof_rows = img.shape[0] - (patch_size - 1) 
    nof_cols = img.shape[1] - (patch_size - 1)
    
    gap_in_x = math.ceil(nof_rows/nof_patches_in_x)
    gap_in_y = math.ceil(nof_cols/nof_patches_in_y)


    patches = np.zeros((nof_patches_in_x, nof_patches_in_y, patch_size, patch_size, 3))

    for i in range(nof_patches_in_x):
        for j in range(nof_patches_in_y):
            patch_idx = (i * gap_in_x) * nof_cols + (j * gap_in_y)
            try:
                patches[i][j] = extracted_patches[patch_idx]
            except:
                print(i, j, nof_rows, nof_cols, gap_in_x, gap_in_y, patch_idx)

    return patches

def get_img_embs():
    model = network.RedCRFENet(input_shape)
    model.load_weights('../../../EG_on_triplet_loss.h5')
    count = 0
    IU_images = os.listdir(IU_cropped_Images_padded_path)
    for img_name in IU_images:
        img_path = os.path.join(IU_cropped_Images_padded_path, img_name)

        img_name = img_name.split('.')[0] + '.npy'
        save_path = os.path.join(saved_embs_path, img_name)

        if not(os.path.exists(save_path)):
            img = cv2.imread(img_path)
            img = img/255.0
            patches = get_patches(img)
            emb_vectors = np.zeros((total_nof_patches, 512))
            k = 0
            for i in range(nof_patches_in_x):
                for j in range(nof_patches_in_y):
                    patch = patches[i][j]
                    patch = patch[np.newaxis,:,:,:]
                    patch = patch.astype(np.float32)
                    emb_vector = model(patch)
                    emb_vectors[k,:] = emb_vector[0,:]
                    k += 1
            
            np.save(save_path, emb_vectors)
            a = np.load(save_path)

            print(count)
            count += 1



############################################### Arguments ##############################################
IU_cropped_Images_padded_path = '/home/ada/Preethi/XRay_Report_Generation/Data/IU_cropped_Images_padded/'
saved_embs_path = "/home/ada/Preethi/XRay_Report_Generation/Code/Normal_vs_Abnormal/siamese/Testing_with_IU_Data/Get_img_embs_from_siamese/Saved_embs_16_512"
patch_size = 128
nof_patches_in_x = 4
nof_patches_in_y = 4
total_nof_patches = nof_patches_in_x * nof_patches_in_y
input_shape = (patch_size, patch_size, 3)
emb_vectors_shape = (nof_patches_in_x*nof_patches_in_y, 512)
############################################### Arguments ##############################################
get_img_embs()



