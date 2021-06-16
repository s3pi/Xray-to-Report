import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, Activation, add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import cv2
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import shuffle
import sys
sys.path.insert(1, '../../Networks')
from CRFENet import make_model
from MultiHead import Narrow_MultiHeadAttention


def tag_img_report_data2(data_folder_report, data_folder_tag, data_folder_image):
  files = os.listdir(data_folder_report)
  files.sort()
  files, test_files = train_test_split(files, test_size = 500, random_state = 666)
  train_files, val_files = train_test_split(files, test_size = 500, random_state = 666)
  train_data = []
  train_labels = []
  for file in train_files:
    current_tag = np.load(os.path.join(data_folder_tag,file))
    train_labels.append(current_tag)
    img_file_name = file.split('.')[0]+'.png'
    current_img = cv2.imread(os.path.join(data_folder_image,img_file_name))
    current_img = current_img.astype(np.float32)
    current_img = current_img / 255.0
    train_data.append(current_img)
  
  val_data = []
  val_labels = []
  for file in val_files:
    current_tag = np.load(os.path.join(data_folder_tag,file))
    val_labels.append(current_tag)
    img_file_name = file.split('.')[0]+'.png'
    current_img = cv2.imread(os.path.join(data_folder_image,img_file_name))
    current_img = current_img.astype(np.float32)
    current_img = current_img/255.0
    val_data.append(current_img)

  test_data = []
  test_labels = []
  for file in test_files:
    current_tag = np.load(os.path.join(data_folder_tag,file))
    test_labels.append(current_tag)
    img_file_name = file.split('.')[0]+'.png'
    current_img = cv2.imread(os.path.join(data_folder_image,img_file_name))
    current_img = current_img.astype(np.float32)
    current_img = current_img / 255.0
    test_data.append(current_img)
  train_data = np.asarray(train_data)
  train_labels = np.asarray(train_labels)
  val_data = np.asarray(val_data)
  val_labels = np.asarray(val_labels)
  test_data = np.asarray(test_data)
  test_labels = np.asarray(test_labels)
  return train_data, train_labels, val_data, val_labels, test_data, test_labels

def Lesp(ip):
  # ip: (2, 238)
  y_pred, y_true = tf.unstack(ip, axis = 0) # (238,)
  
  neg_mask  = tf.where(tf.equal(y_true, 0), tf.ones_like(y_pred), tf.zeros_like(y_pred)) # (238,)

  # Get positive and negative scores   
  positives = tf.boolean_mask(y_pred,y_true)
  negatives = tf.boolean_mask(y_pred,neg_mask)

  negatives = tf.expand_dims(negatives, 1)
  positives = tf.expand_dims(positives, 0)

  difference = negatives - positives

  exp_difference = tf.math.exp(difference)

  total_difference = tf.reduce_sum(exp_difference)

  total_difference = tf.constant(1, dtype = tf.float32) + total_difference

  loss_temp = tf.math.log(total_difference)
  return loss_temp


def lesp_loss(target, output):
  # target: bs, 238
  # output: bs, 238
  combine_tensor = tf.stack([output, target], axis = 1)  # bs, 2, 238
  loss_tensor = tf.map_fn(Lesp, combine_tensor, dtype = tf.float32) # (bs,)
  loss = tf.math.reduce_mean(loss_tensor) # (1,)
  return loss


def make_tc_model():
  base_model = make_model((313, 306, 3))
  base_model.load_weights('../../../Model_Weights/CRFENET_NIH.h5')
  ip = Input((5,313, 306, 3))
  base_op_list = []
  for i in range(5):
  	base_op = base_model(ip[:,i,:,:,:])
  	base_op = GlobalAveragePooling2D()(base_op)
  	base_op = tf.expand_dims(base_op,1)
  	base_op_list.append(base_op)
  	
  op = tf.keras.layers.Concatenate(1)(base_op_list)

  op,attention = Narrow_MultiHeadAttention(256,8)(op,op,op)
  op = tf.math.reduce_mean(op, axis = 1)
  op = Dense(237, activation = 'sigmoid')(op)
  final_model = Model(ip,op)
  final_model.summary()
  base_model.trainable = False
  final_model.compile(loss = lesp_loss, optimizer = tf.keras.optimizers.Adam(1e-5))
  return final_model, base_model

def train():
    # global train_data, train_labels, val_data, val_labels
    model, base_model= make_tc_model()
    model.summary()
    save_model_list=[]
    test_e = 0
    min_loss = 100.0
    batch_count = 0
    best_acc = 0.0
    num_of_batches = int(math.ceil(train_data.shape[0]/batch_size))

    for e in range(num_epochs):
        train_data, train_labels = shuffle(train_data, train_labels, random_state = 666)
        per_epoch_train_loss = 0.0

        for batch_num in range(num_of_batches):
            batch_X_train = np.zeros((batch_size, 313, 306, 3))
            batch_y_train = np.zeros((batch_size, 238))

            b = 0
            for j in range(batch_num*batch_size, min((batch_num+1)*batch_size, train_data.shape[0])):
                batch_X_train[b, :, :, :] = train_data[j,:,:,:]
                batch_y_train[b,:] = train_labels[j,:]
                b+=1

            per_batch_train_loss = model.train_on_batch(batch_X_train, batch_y_train)
            per_epoch_train_loss += per_batch_train_loss
            print('epoch_num: %d, batch_num: %d, loss: %f\n' % (e, batch_num, per_batch_train_loss))
            if batch_count == 50:
                batch_count = 0
                per_epoch_test_loss = 0.0
                for i in range(val_data.shape[0]):
                    err = model.test_on_batch(val_data[i:i+1,:,:,:], val_labels[i:i+1,:])
                    per_epoch_test_loss += err
                per_epoch_test_loss /= val_data.shape[0]

                test_e+=1
                print('---------------------------------------------------------------------\n')
                print('test_num: %d, \nloss: %f\n' % (test_e, per_epoch_test_loss))
                
                # if per_epoch_test_pos_acc >= max_acc_pos and per_epoch_test_neg_acc >= max_acc_neg:
                # if (per_epoch_test_pos_acc + per_epoch_test_neg_acc) > (max_acc_pos + max_acc_neg):
                if per_epoch_test_loss <= min_loss:
                    save_model_list.append(test_e)
                    model.save_weights(os.path.join(model_weights_path, 'model_' + str(test_e)+'.h5'))
                    base_model.save_weights(os.path.join(model_weights_path, 'base_model_' + str(test_e)+'.h5'))
                    if len(save_model_list)>10:
                        del_model_count = save_model_list.pop(0)
                        os.remove(os.path.join(model_weights_path, 'model_' + str(del_model_count)+'.h5'))
                        os.remove(os.path.join(model_weights_path, 'base_model_' + str(del_model_count)+'.h5'))

                    print('Model loss improved from %f to %f\nSaving Weights\n'%(min_loss, per_epoch_test_loss))
                    min_loss = per_epoch_test_loss
                    
                else:
                    print('Best loss: %f\n'%(min_loss))
                print('----------------------------------------------------------------------\n')
            batch_count += 1
        per_epoch_train_loss /=num_of_batches



# def test(): # For Ablation Study
#     global test_data, test_labels
#     model, base_model= make_model()
#     model.load_weights(os.path.join(Densenet_model_weight_path),'model_1079.h5')
#     current_tag = test_labels[1:2,:]
#     print(current_tag[0,:])
#     for i in range(current_tag.shape[1]):
#         if current_tag[0,i] == 1:
#             print(i,endl=' ')


#################################### Global Arguments #######################################
batch_size = 32
num_epochs = 10000
weight_decay = 1e-5
#############################################################
server_path = "/mnt/Data1/Preethi/XRay_Report_Generation"
# server_path = "/home/ada/Preethi/XRay_Report_Generation"
#############################################################
IU_7430_data_details_path = os.path.join(server_path, 'Data', 'text_data_img_caption') # IU data has 7430 xray reports. Each report is a seq of len 260.
tags_folder_path = os.path.join(server_path, 'Data', 'tag_data_automatic_trim')
image_folder_path = os.path.join(server_path, 'Data', 'IU_cropped_Images_padded') # IU images that are cropped and padded to same size (313 x 306).
result_files_path = os.path.join(server_path, 'Code', '3_tag_prediction', 'Results_ranking_custom_512_2')
model_weights_path = os.path.join(server_path, 'Code', '3_tag_prediction', 'Model_Weights_ranking_custom_512_2')
Densenet_model_weight_path = os.path.join(server_path, 'Code', 'base_model_step1_42.h5')
#################################### Global Arguments #######################################
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
if not os.path.exists(result_files_path):
    os.makedirs(result_files_path)

train_data, train_labels, val_data, val_labels, _, _ = tag_img_report_data2(IU_7430_data_details_path, tags_folder_path, image_folder_path)
train()
# _,_,_,_,test_data, test_labels,class_weights = tag_img_report_data2(IU_7430_data_details_path, tags_folder_path, image_folder_path)
# test()