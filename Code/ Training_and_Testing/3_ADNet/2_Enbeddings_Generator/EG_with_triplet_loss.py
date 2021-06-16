## Read Me ##
# Weights of RedCRFENet in RedCRFENet+Dense(1) trained on ROIs of NIHCC data can be found in ../../Networks/RedCRFENet_on_ROI_patches.py.
# The above weights are loaded on RedCRFENet, followed by a couple of layers and is trained on ROIs of NIHCC data using triplet loss and best model weights are named as "EG_on_triplet_loss.h5" and saved.

import distance_net as dn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense
import sys
import os
import importlib
import argparse
import time
import numpy as np
import random

import sys
sys.path.insert(1, '../../Networks') 
import model_siamese_RedCRFENet as network

def main(args):
    # network = importlib.import_module(args.model_def)
    subdir = 'patch_128'
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    normal_image_paths, abnormal_image_paths, abnormal_ROIs = dn.get_dataset(args.data_dir, args.data_csv_path, args.box_csv_path) # List of image class objects

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    input_shape = (args.patch_size, args.patch_size, 3)
    base_model = network.RedCRFENet(input_shape)
    base_model.load_weights('../../../RedCRFENet_on_ROI_patches.h5') #classifier_model_weights as mentioned in the ADNet_readme.txt
    
    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)
    # margin = Input(shape=(1,))

    anchor_embedding = base_model(anchor_input)
    positive_embedding = base_model(positive_input)
    negative_embedding = base_model(negative_input)
    
    combined_embeddings = Lambda(dn.combine_embeddings)([anchor_embedding,positive_embedding,negative_embedding])
    model = Model([anchor_input,positive_input,negative_input],combined_embeddings)
    model.summary()

    learning_rate = args.learning_rate

    if args.optimizer=='ADAGRAD':
        opt = keras.optimizers.Adagrad(learning_rate)
    elif args.optimizer=='ADADELTA':
        opt = tf.keras.optimizers.Adadelta(learning_rate, rho=0.9, epsilon=1e-6)
    elif args.optimizer=='ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.1)
    elif args.optimizer=='RMSPROP':
        opt = tf.keras.optimizers.RMSprop(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif args.optimizer=='SGD':
        opt = tf.keras.optimizers.SGD(learning_rate, 0.9, nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')

    model.compile(loss=dn.triplet_loss_func,optimizer=opt)

    start_batch_id = 1
    counter_for_margin=0
    counter_for_stopping = 0
    continue_training = True

    epoch = 1
    final_loss=0.0
    count=0.0
    train_time = 0.0
    while continue_training and (epoch <= args.max_nrof_epochs):
        batch_number = start_batch_id
        while batch_number < args.epoch_size:
            # Sample people randomly from the dataset
            # image_paths, num_per_class = dn.sample_people(train_set, args.nof_classes_per_batch, args.nof_abnormal_image_samples)
            image_paths, sample_abnormal_ROIs, num_per_class = dn.sample_images(normal_image_paths, abnormal_image_paths, abnormal_ROIs, args.nof_abnormal_image_samples, args.nof_normal_image_samples)

            print('Running forward pass on sampled images: ', end='')
            start_time = time.time()
            input_images = dn.load_data(image_paths, sample_abnormal_ROIs, args.patch_size)
            emb_array = []
            for i in range(len(image_paths)):
                emb = base_model.predict(input_images[i:i+1,:])
                emb_array.append(emb.reshape((args.embedding_size,)))

            emb_array=np.asarray(emb_array) 

            print('%.3f' % (time.time()-start_time))

            # Select triplets based on the embeddings
            print('Selecting suitable triplets for training')
            triplets, nrof_random_negs, nrof_triplets = dn.select_triplets(emb_array, num_per_class,
                input_images, args.nof_classes_per_batch, args.alpha)
            selection_time = time.time() - start_time
            print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
                (nrof_random_negs, nrof_triplets, selection_time))
            print("Number of triplets "+str(nrof_triplets))

            triplet_anchors =[]
            triplet_positives = []
            triplet_negatives = []
            for i in range(len(triplets)):
                triplet_anchors.append(triplets[i][0])
                triplet_positives.append(triplets[i][1])
                triplet_negatives.append(triplets[i][2])
            
            # Criteria for increasing margin

            if nrof_triplets < 0.1*nrof_random_negs:
                counter_for_margin+=1
            else:
                counter_for_margin = 0

            if (counter_for_margin == 3) and (args.alpha<=args.max_alpha):
                args.alpha+=0.1
                counter_for_margin=0

            # Criteria for early stopping
            if (nrof_triplets < 0.05*nrof_random_negs) and (args.alpha >= args.max_alpha):
                counter_for_stopping+=1
            else:
                counter_for_stopping=0
            if counter_for_stopping == 3:
                continue_training = False

            # Perform training on the selected triplets
            nrof_batches = int(np.ceil(nrof_triplets/args.batch_size))
            i = 0
            for i in range(nrof_batches):
                start_time = time.time()
                anchor_input = triplet_anchors[i*args.batch_size:i*args.batch_size+args.batch_size]
                positive_input = triplet_positives[i*args.batch_size:i*args.batch_size+args.batch_size]
                negative_input = triplet_negatives[i*args.batch_size:i*args.batch_size+args.batch_size]
                anchor_input = np.asarray(anchor_input)
                positive_input = np.asarray(positive_input)
                negative_input = np.asarray(negative_input)
                margin_input = [args.alpha]*anchor_input.shape[0]
            
                margin_input = np.asarray(margin_input)
                margin_input = np.expand_dims(margin_input,1)
                margin_input = np.expand_dims(margin_input,1)  
             
                err = model.train_on_batch([anchor_input,positive_input,negative_input], margin_input)
                duration = time.time() - start_time
                # print(emb[0,:])
                print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %f\tMargin %f' %
                      (epoch, batch_number+1, args.epoch_size, duration, err, args.alpha))
                final_loss+=err
                count+=1
                if count==25:
                    with open(log_dir+'/training.log','a') as f:
                        f.write(str(final_loss/count)+'\t'+str(nrof_triplets)+'\t'+str(args.alpha)+'\n')
                    final_loss=0.0
                    count=0
                batch_number+=1
                train_time += duration
        epoch+=1
        # temp_model.trainable = True
        base_model.save_weights(model_dir+'/saved_model_'+str(epoch%10)+'.h5') # best model weights named "EG_on_triplet_loss.h5" is saved after training.
        # temp_model.trainable = False
        # final_model.save_weights(model_dir+'/saved_model_final_'+str(epoch%10)+'.h5')

    return train_time

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='./logs/siamese')
    # parser.add_argument('--models_base_dir', type=str,
    #     help='Directory where to write trained models and checkpoints.', default='./models/siamese')
    # parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the model.', default='model_siamese_RedCRFENet')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='/home/ada/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/Images_410/')
    parser.add_argument('--data_csv_path', type=str,
        help='Path to the data csv containing label information.',
        default='/home/ada/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/Data_Entry_2017.csv')
    parser.add_argument('--box_csv_path', type=str,
        help='Path to the box csv containing ROI information.',
        default='/home/ada/Preethi/XRay_Report_Generation/Data/ChestXray-NIHCC/BBox_List_2017.csv')
    parser.add_argument('--drop_probability', type=float,
        help='Dropping probability of dropout for the fully connected layer(s).', default=0.5)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=512)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=1e-5)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'SGD'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=1e-5)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--nof_classes_per_batch', type=int,
        help='Number of people per batch.', default=2) # number of classes per batch
    parser.add_argument('--nof_abnormal_image_samples', type=int,
        help='Number of images for abnormal class.', default=30) # number of random abnormal images selected per class no each iteration
    parser.add_argument('--nof_normal_image_samples', type=int,
        help='Number of images for normal class.', default=30) # number of random normal images selected per class no each iteration
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=1000)
    parser.add_argument('--alpha', type=float,
        help='Initial positive to negative triplet distance margin.', default=1.0)
    parser.add_argument('--max_alpha', type=float,
        help='Maximum positive to negative triplet distance margin.', default=2.5)
    parser.add_argument('--batch_size', type=int,
        help='Number of triplets to process in a batch.', default=32)
    parser.add_argument('--patch_size', type=int,
        help='Number of triplets to process in a batch.', default=128)
    return parser.parse_args(argv)

if __name__ == '__main__':
    train_time=main(parse_arguments(sys.argv[1:]))
    print('Total Training time is '+str(train_time)+' seconds')





