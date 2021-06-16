## Read Me ##
# IU data (all the data - train and test) is cropped to same size and its features are extracted from CRFENet that is trained on NIHCC data.
# Weights used are CRFENET_NIH.h5
# 10x10x512 size features are saved in the hard disk.

from tensorflow import keras
import cv2
import numpy as np
import os

import sys
sys.path.insert(1, '../Networks') 
from CRFEENET import make_model     # Depth separable conv layers of CREFENet.

ip_folder = '/mnt/Data1/Preethi/XRay_Report_Generation/Data/IU_cropped_Images_padded/'
op_folder = '/mnt/Data1/Preethi/XRay_Report_Generation/Data/IU_features_512/'

base_model = make_model((313,306,3))
# base_model = keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=None, input_shape=(313, 306, 3), pooling=None, classes=1000)
base_model.load_weights('../../Model_Weights/CRFENET_NIH.h5')   # 1_train_CRFENet_on_NIHCC.py produces this weight file. 

for file in os.listdir(ip_folder):
    current_img = cv2.imread(ip_folder+file)
    current_img = current_img.astype(dtype=np.float32)
    img = current_img / 255.0
    img = np.expand_dims(img,axis=0)
    feat = base_model.predict(img)
    feat = np.squeeze(feat)
    np.save(op_folder+file.split('.')[0],feat)  # Features of IU data predicted from CREFENet_trained_on_NIH is saved.
