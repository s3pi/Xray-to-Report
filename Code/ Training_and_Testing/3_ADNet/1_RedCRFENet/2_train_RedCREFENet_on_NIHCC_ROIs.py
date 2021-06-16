## Read Me ##
# Weights of RedCRFENet+Dense(15) trained on NIHCC data can be found in ../../Networks/RedCRFENet.py.
# Another model RedCRFENet+Dense(1) for binary classification between normal and abnormal patches is trained using ROIs of NIHCC data is done and weights are saved as RedCRFENet_on_ROI_patches.h5

import distance_net as dn
import importlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import random
import numpy as np

import sys
sys.path.insert(1, '../../Networks') 
import RedCRFENet

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, SeparableConv2D, add, Conv3D, Conv2D, MaxPool3D, MaxPooling2D, GlobalAveragePooling3D, GlobalAveragePooling2D, Dropout, Dense, Lambda, TimeDistributed, LSTM, Bidirectional, GlobalAveragePooling1D, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation


data_csv_path = '/home/ada/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/Data_Entry_2017.csv'
data_dir = '/home/ada/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/Images_410/'
box_csv_path = '/home/ada/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/BBox_List_2017.csv'
normal_image_paths, abnormal_image_paths, abnormal_ROIs = dn.get_dataset(data_dir, data_csv_path, box_csv_path)
weights_paths = '../../../Model_Weights/RedCRFENet.h5' # 1_train_RedCRFENet_on_NIHCC.py produces this weight file. 

patch_size = 128
input_shape = (patch_size, patch_size, 3)
base_model = RedCRFENet.make_model(input_shape)
input_tensor = Input((input_shape))
op = base_model(input_tensor)
op = GlobalAveragePooling2D()(op)
temp_op = Dense(15)(op)
temp_model = Model(input_tensor,temp_op)
temp_model.load_weights(weights_paths)
op = Dense(1, activation='sigmoid')(op)
model = Model(input_tensor, op)
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(1e-5), metrics = ['accuracy'])

for epoch in range(1000):
    for batch in range(100):
        # num_abnormal = random.randint(80,120)
        num_abnormal = 100
        image_paths, sample_abnormal_ROIs, num_per_class = dn.sample_images(normal_image_paths, abnormal_image_paths, abnormal_ROIs, num_abnormal, 200 - num_abnormal)
        labels = np.zeros((200,1), dtype = np.float32)
        labels[:num_abnormal,:] = 1.0
        # labels[100:,1] = 1.0
        input_images = dn.load_data(image_paths, sample_abnormal_ROIs, patch_size)
        input_images, labels = shuffle(input_images, labels)
        his = model.fit(input_images, labels, batch_size = 32, epochs = 1)
        base_model.save_weights('RedCRFENet_on_ROI_patches.h5')
        if his.history['accuracy'][0] > 0.85:
            exit()

