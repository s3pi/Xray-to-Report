## Read Me ##
# Saved 16x512 features (per image of IU data) from 1_get_img_embs_from_EG.py is trained CNN network made for classifying the IU data between normal and abnormal.
# Same network is used for testing. 

import importlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, SeparableConv2D, add, Conv3D, Conv2D, MaxPool3D, MaxPooling2D, GlobalAveragePooling3D, GlobalAveragePooling2D, Dropout, Dense, Lambda, TimeDistributed, LSTM, Bidirectional, GlobalAveragePooling1D, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation

from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Dropout
import os
import math
from sklearn.utils import shuffle
import nltk
import cv2

def make_model():
    input_tensor = Input(input_shape)
    
    op = Conv2D(512, 3, padding = 'same')(input_tensor)
    # op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Activation('tanh')(op)

    op = Conv2D(512, 4, padding = 'valid')(op)
    # op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Activation('tanh')(op)
    
    op = GlobalAveragePooling2D()(op)
    op = Dense(1, activation='sigmoid')(op)

    model = Model(input_tensor, op)
    model.summary()
    return model

def open_metric_files():
    per_batch_train_metrics_file = open(result_files_path + '/training_per_batch_metrics' + '.txt', 'a')
    per_epoch_train_metrics_file = open(result_files_path + '/training_per_epoch_metrics' + '.txt', 'a')
    per_epoch_val_metrics_file = open(result_files_path + '/val_metrics' + '.txt', 'a')

    return per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file

def write_metric_files(files, testues):
    for i in range(len(files)):
        files[i].write(str(testues[i])+ '\n')

def load_inp_to_model(path):
    data = []
    labels = []
    tags_dict = dict()
    with open(tags_path, 'r') as g:
        a = g.readlines()
        for each in a:
            each_list = each.split()
            tags_dict[each_list[0]] = int(each_list[-1])

    with open(path, 'r') as f:
        a = f.readlines()
        num_of_files = len(a)
        for file_name in a:
            file_name = file_name.split()[0]
            temp = file_name + '.npy'
            file_path = os.path.join(saved_embs_path, temp)
            assert os.path.exists(file_path)
            data.append(file_path)
            labels.append(tags_dict[file_name])

    labels = np.asarray(labels)
    labels.astype(np.float32)
    return data, labels

def load_data(mode):
    if mode is 'train':
        
        train_filenames_txt_path = os.path.join(data_path, 'train_list.txt')
        test_filenames_txt_path = os.path.join(data_path, 'val_list.txt')
        
        train_data_paths, train_labels = load_inp_to_model(train_filenames_txt_path) 
        test_data_paths, test_labels = load_inp_to_model(test_filenames_txt_path)

        return train_data_paths, train_labels, test_data_paths, test_labels

    if mode is 'test':
        test_filenames_txt_path = os.path.join(data_path, 'test_list.txt')
        test_data_paths, test_labels = load_inp_to_model(test_filenames_txt_path) 
        
        return test_data_paths ,test_labels

def train():
    train_data_paths, train_labels, val_data_paths, val_labels = load_data('train')
    min_loss = 100.0
    save_model_list = []
    num_of_batches = int(math.ceil(len(train_data_paths)/batch_size))

    test_e = 0
    for e in range(num_epochs):
        per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file = open_metric_files()
        train_data_paths, train_labels = shuffle(train_data_paths, train_labels)
        per_epoch_train_loss = 0.0
        per_epoch_train_acc = 0.0

        for batch_num in range(num_of_batches):
            batch_X = np.zeros((batch_size,) + (input_shape))
            batch_Y = np.zeros((batch_size, 1))

            b = 0
            for j in range(batch_num*batch_size, min((batch_num+1)*batch_size, len(train_data_paths))):
                temp = np.reshape(np.load(train_data_paths[j]), input_shape)
                batch_X[b,:,:,:] = temp
                batch_Y[b,:] = train_labels[j]
                b += 1

            per_batch_train_loss, per_batch_train_acc = model.train_on_batch(batch_X, batch_Y)
            
            print('epoch_num: %d, batch_num: %d, loss: %f, class_wise_accuracy: %s\n' % (e, batch_num, per_batch_train_loss, per_batch_train_acc))

            write_metric_files([per_batch_train_metrics_file], [[e, batch_num, per_batch_train_loss, per_batch_train_acc]])

            per_epoch_train_loss += per_batch_train_loss
            per_epoch_train_acc += per_batch_train_acc

        per_epoch_train_loss = per_epoch_train_loss / num_of_batches
        per_epoch_train_acc = per_epoch_train_acc / num_of_batches
        write_metric_files([per_epoch_train_metrics_file], [[e, per_epoch_train_loss, per_epoch_train_acc]])

        per_epoch_val_loss, per_epoch_val_acc = 0.0, 0.0
        for i in range(len(val_data_paths)):
        # for i in range(2):
            val_inp = np.reshape(np.load(val_data_paths[i]), (input_shape))
            val_Y_labels = val_labels[i]

            val_data = val_inp[np.newaxis,:,:,:]
            val_Y_labels = np.expand_dims(val_Y_labels, 0)

            curr_val_loss, curr_val_acc = model.test_on_batch(val_data, val_Y_labels)

            per_epoch_val_loss += curr_val_loss
            per_epoch_val_acc += curr_val_acc

        per_epoch_val_loss /= len(val_data_paths)
        per_epoch_val_acc /= len(val_data_paths)

        test_e+=1
        print('---------------------------------------------------------------------\n')
        print('val_num: %d, loss: %f, \nclass_wise_accuracy: %s\n' % (test_e, per_epoch_val_loss, per_epoch_val_acc))

        if per_epoch_val_loss < min_loss:
            save_model_list.append(test_e)
            # base_model.trainable = True
            # base_model.save_weights(model_weights_path + '/base_model_' + str(test_e)+'.h5') 
            model.save_weights(model_weights_path + '/model_' + str(test_e)+'.h5') # Best model weights of CNN after EG is saved as "CNN_after_EG.h5"
            # base_model.trainable = False
            if len(save_model_list)>max_models_to_save:
                del_model_count = save_model_list.pop(0)
                os.remove(model_weights_path + '/model_' + str(del_model_count)+'.h5')
                # os.remove(model_weights_path + '/base_model_' + str(del_model_count)+'.h5')
                
            print('Model loss improved from %f to %f. Saving Weights\n'%(min_loss, per_epoch_val_loss))
            min_loss = per_epoch_val_loss
        else:
            print('Best loss: %f\n'%(min_loss))

        print('----------------------------------------------------------------------\n')
        write_metric_files([per_epoch_val_metrics_file], [[min_loss, test_e, per_epoch_val_loss, per_epoch_val_acc]])
        
        per_batch_train_metrics_file.close() 
        per_epoch_train_metrics_file.close() 
        per_epoch_val_metrics_file.close()

def test():
    # model = make_model()
    model.load_weights("'../../../CNN_after_EG.h5")
    # model.compile(loss='binary_crossentropy',optimizer=Adam(1e-5), metrics = ['accuracy'])
    test_data_paths, test_labels = load_data('test')
    per_epoch_test_metrics_file = open(result_files_path + '/test_metrics' + '.txt', 'a')

    per_epoch_test_loss, per_epoch_test_acc = 0.0, 0.0
    for i in range(len(test_data_paths)):
    # for i in range(2):
        test_inp = np.reshape(np.load(test_data_paths[i]), input_shape)
        test_Y_labels = test_labels[i]

        test_data = test_inp[np.newaxis,:,:,:]
        test_Y_labels = np.expand_dims(test_Y_labels, 0)

        curr_test_loss, curr_test_acc = model.test_on_batch(test_data, test_Y_labels)

        per_epoch_test_loss += curr_test_loss
        per_epoch_test_acc += curr_test_acc

    per_epoch_test_loss /= len(test_data_paths)
    per_epoch_test_acc /= len(test_data_paths)

    print('loss: %f, \nclass_wise_accuracy: %s\n' % (per_epoch_test_loss, per_epoch_test_acc))
    write_metric_files([per_epoch_test_metrics_file], [[per_epoch_test_loss, per_epoch_test_acc]])
    per_epoch_test_metrics_file.close()

############################################### Arguments ##############################################
# IU_cropped_Images_padded_path = '/home/ada/Preethi/XRay_Report_Generation/Data/IU_cropped_Images_padded/'
data_path = '/home/ada/Preethi/XRay_Report_Generation/Data'
tags_path = '/home/ada/Preethi/XRay_Report_Generation/Data/tags_automatic.txt'
saved_embs_path = "/home/ada/Preethi/XRay_Report_Generation/Code/Normal_vs_Abnormal/siamese/Testing_with_IU_Data/Get_img_embs_from_siamese/Saved_embs_16_512"
model_weights_path = '/home/ada/Preethi/XRay_Report_Generation/Code/Normal_vs_Abnormal/siamese/Testing_with_IU_Data/Train_embs_with_cnn/Model_Weights_bn'
result_files_path = '/home/ada/Preethi/XRay_Report_Generation/Code/Normal_vs_Abnormal/siamese/Testing_with_IU_Data/Train_embs_with_cnn/Results_bn'

num_epochs = 10000
batch_size = 128
max_models_to_save = 2
############################################### Arguments ##############################################
input_shape = (4, 4, 512)
model = make_model()
model.compile(loss='binary_crossentropy',optimizer=Adam(1e-5), metrics = ['accuracy'])
# train()
test()



