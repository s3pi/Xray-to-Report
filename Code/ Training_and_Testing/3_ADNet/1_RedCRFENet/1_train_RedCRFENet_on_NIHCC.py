## Read Me ##
# RedCRFENet+Dense(15) is trained on NIHCC data and weights can be found in ../../Networks/RedCRFENet.py.

import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tensorflow import keras
from sklearn.utils import shuffle
import csv

import sys
sys.path.insert(1, '../../Networks') 
from RedCRFENet import make_model

def name_numbers(length, number):
    return '0' * (length - len(str(number))) + str(number)

def calc_acc(y_true,y_pred):
    acc = np.zeros((16,),dtype=np.float32)
    for i in range(15):
        for j in range(y_true.shape[0]):
            if ((y_true[j,i] >= 0.5) and (y_pred[j,i]>= 0.5)) or ((y_true[j,i] < 0.5) and (y_pred[j,i]< 0.5)):
                acc[i]+=1
    for i in range(y_true.shape[0]):
        count = 0
        for j in range(15):
            if ((y_true[i,j] >= 0.5) and (y_pred[i,j]>= 0.5)) or ((y_true[i,j] < 0.5) and (y_pred[i,j]< 0.5)):
                count +=1
        if count == 15:
            acc[15]+=1
    acc = acc / y_true.shape[0]
    return acc


def make_disease_list():
    with open(csv_path, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)
        unique_class_names = []
        for row in csvreader:
            unique_class_names.extend(row[1].split('|'))

    unique_class_names = sorted(list(set(unique_class_names)))
    
    return unique_class_names

def make_dict_img_label():
    with open(csv_path, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader)
        dict_img_label = {}
        for row in csvreader:
            dict_img_label[row[0]] = row[1].split('|')

    return dict_img_label

def data_loader():
    img_path = []
    labels = []
    unique_class_names = make_disease_list()
    print(unique_class_names)
    dict_img_label = make_dict_img_label()

    for i in range(1, 13):
        name = name_numbers(3, i)
        current_path = data_path + "/images_" + str(name)
        for file in os.listdir(current_path):
            img_path.append(os.path.join(current_path, file))
            classes = dict_img_label[file]
            one_hot_label = np.zeros(15, dtype = 'int')
            for each in classes:
                one_hot_label[unique_class_names.index(each)] = 1

            labels.append(one_hot_label)

    img_path = np.asarray(img_path)
    labels = np.asarray(labels)
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(img_path, labels, test_size = 0.2, random_state = 666)
    print(X_train_paths.shape, X_test_paths.shape, y_train.shape, y_test.shape)
    print(X_train_paths[100], y_train[100])
    return X_train_paths, X_test_paths, y_train, y_test

def open_metric_files():
    per_batch_metrics_file = open(result_files_path + '/training_per_batch_metrics' + '.txt', 'a')
    per_epoch_metrics_file = open(result_files_path + '/training_per_epoch_metrics' + '.txt', 'a')
    test_metrics_file = open(result_files_path + '/test_metrics' + '.txt', 'a')

    return per_batch_metrics_file, per_epoch_metrics_file, test_metrics_file

def write_metric_files(files, values):
    for i in range(len(files)):
        files[i].write(str(values[i])+ '\n')

def close_metric_files(files):
    for each_file in files:
        each_file.close()

def save_models(prev_epoch, e):
    if prev_epoch == -1:
        model.save_weights(model_weights_path + "/model_last_but_1.h5")
    elif e == 1:
        model.save_weights(model_weights_path + "/model_last.h5")
    elif (e // save_every == 0) or (e == num_epochs - 1):
        os.rename(model_weights_path + "/model_last.h5", model_weights_path + "/model_last_but_1.h5")
        model.save_weights(model_weights_path + "/model_last.h5")

def load_from_paths(X_test_paths):
    X_test = np.zeros((len(X_test_paths), size, size, 3))
    for i in range(len(X_test_paths)):
        X_test[i] = cv2.imread(X_test_paths[i])

    return X_test

def evaluate(X_test, y_test):
    # This function in case, we wanna run the evaluation on various batches to deal with memory issue.
    score = model.evaluate(X_test, y_test)
    return score

def train():
    X_train_paths, X_test_paths, y_train, y_test = data_loader() #NIH data
    base_model.summary() #RedCRFENet
    batch_count = 0

    test_e = 0
    prev_epoch = -1
    min_loss = 100.0
    save_model_list = []
    for e in range(num_epochs):
        X_train_paths, y_train = shuffle(X_train_paths, y_train, random_state = 2)
        per_batch_metrics_file, per_epoch_metrics_file, test_metrics_file = open_metric_files()
        per_epoch_loss = 0.0
        per_epoch_accuracy = np.zeros((16,),dtype=np.float32)
        
        num_of_batches = int(len(X_train_paths)/batch_size)

        for batch_num in range(num_of_batches):
            batch_X_train = np.zeros((batch_size, size, size, 3))
            batch_y_train = np.zeros((batch_size, 15))
            b = 0

            for j in range(batch_num*batch_size, min((batch_num+1)*batch_size, len(X_train_paths))):
                img = cv2.imread(X_train_paths[j])
                batch_X_train[b, :, :] = img        
                batch_y_train[b,:] = y_train[j,:]
                b += 1
            batch_X_train = batch_X_train/255.0
            loss = model.train_on_batch(batch_X_train, batch_y_train)
            y_pred = model.predict(batch_X_train)
            accuracy = calc_acc(batch_y_train,y_pred)
            batch_count+=1
            print('epoch_num: %d, batch_num: %d, loss: %f, hard_accuracy: %f\nclass_wise_accuracy: %s\n' % (e, batch_num, loss,  accuracy[15], accuracy[:15]))
            write_metric_files([per_batch_metrics_file], [[e, batch_num, loss, accuracy]])

            per_epoch_loss += loss
            per_epoch_accuracy += accuracy

            if batch_count == 100:
                batch_count = 0
                per_epoch_test_loss = 0.0
                per_epoch_test_acc = np.zeros((16,),dtype=np.float32)
                num_of_testing_batches = int(len(X_test_paths)/batch_size)
                for test_batch_num in range(num_of_testing_batches):
                # for test_batch_num in range(10):
                    batch_X_test = np.zeros((batch_size, size, size, 3))
                    batch_y_test = np.zeros((batch_size, 15))
                    b_test = 0

                    for j in range(test_batch_num*batch_size, min((test_batch_num+1)*batch_size, len(X_test_paths))):
                        img = cv2.imread(X_test_paths[j])
                        batch_X_test[b_test, :, :] = img        
                        batch_y_test[b_test,:] = y_test[j,:]
                        b_test += 1
                    batch_X_test = batch_X_test/255.0
                    test_loss= model.test_on_batch(batch_X_test, batch_y_test)
                    y_pred = model.predict(batch_X_test)
                    test_accuracy = calc_acc(batch_y_test,y_pred)

                    per_epoch_test_loss += test_loss
                    per_epoch_test_acc += test_accuracy

                per_epoch_test_loss = per_epoch_test_loss / num_of_testing_batches
                
                per_epoch_test_acc = per_epoch_test_acc / num_of_testing_batches
                
                print('---------------------------------------------------------------------\n')
                print('test_num: %d, loss: %f, hard_accuracy: %f\nclass_wise_accuracy: %s\n' % (test_e, per_epoch_test_loss, per_epoch_test_acc[15],per_epoch_test_acc[:15]))
                write_metric_files([test_metrics_file], [[test_e, per_epoch_test_loss, per_epoch_test_acc]])
                

                if per_epoch_test_loss < min_loss:
                    save_model_list.append(test_e)
                    # base_model.trainable = True
                    base_model.save_weights(model_weights_path + '/base_model_' + str(test_e)+'.h5')
                    model.save_weights(model_weights_path + '/model_' + str(test_e)+'.h5') # best model weight is RedCRFENET_NIH.h5 and is saved in Model_Weights.

                    if len(save_model_list) > 10:
                        del_index = save_model_list.pop(0)
                        os.remove(model_weights_path+'/base_model_'+str(del_index)+'.h5')
                        os.remove(model_weights_path+'/model_'+str(del_index)+'.h5')
                    # prev_epoch = 0
                    print('Model loss improved from %f to %f. Saving Weights\n'%(min_loss,per_epoch_test_loss))
                    min_loss = per_epoch_test_loss
                else:
                    print('Best loss: %f\n'%(min_loss))
                print('---------------------------------------------------------------------\n')
                test_e += 1

        per_epoch_loss = per_epoch_loss / num_of_batches
        per_epoch_accuracy = per_epoch_accuracy / num_of_batches
        write_metric_files([per_epoch_metrics_file], [[e, per_epoch_loss, per_epoch_accuracy]])
        close_metric_files([per_batch_metrics_file, per_epoch_metrics_file, test_metrics_file])

def custom_base_model_training(): #RedCRFFENet is the base model. This model puts Dense(15) on top of it.
    base_model = make_model((410, 410, 3)) 
    input_tensor = Input((410, 410, 3))
    output_tensor = base_model(input_tensor)
    op = GlobalAveragePooling2D()(output_tensor)
    op = Dense(15, activation = "sigmoid")(op)
    model = Model(input_tensor, op) #model is CRFENet+Dense(15) to classify NIH data
    adam = Adam(lr = 1e-4)
    model.compile(loss = "binary_crossentropy", optimizer = adam)

    return model, base_model

###############################################################################################
data_path = "/mnt/Data1/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/Images_410"
result_files_path = "/mnt/Data1/Preethi/XRay_Report_Generation/Code/Results_custom_siam_1024"
model_weights_path = "/mnt/Data1/Preethi/XRay_Report_Generation/Code/Model_Weights_custom_siam_1024"
csv_path = "/mnt/Data1/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/Data_Entry_2017.csv"

if not os.path.exists(result_files_path):
    os.makedirs(result_files_path)
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
model, base_model = custom_base_model_training() 
batch_size = 32
num_epochs = 10000
save_every = 1
size = 410
###############################################################################################
train()

