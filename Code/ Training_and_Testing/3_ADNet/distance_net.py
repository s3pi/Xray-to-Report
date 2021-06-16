import os
import tensorflow as tf
import numpy as np
import cv2
import csv
import random

def select_triplets(embeddings, nrof_images_per_class, input_images, nof_classes_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    final_counter_sax=0
    
    for i in range(nof_classes_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1) # (300, 1)
            for pair in range(j, nrof_images): # For every possible positive pair.
                final_counter_sax+=1
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<=neg_dists_sqr))[0]  # FaceNet selection
                # all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((input_images[a_idx,:,:,:], input_images[p_idx, :,:,:], input_images[n_idx,:,:,:]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets) #24000, 3 (paths)

    return triplets, num_trips, len(triplets)

def load_data(image_paths, sample_abnormal_ROIs, patch_size):
    no_of_images = len(image_paths)
    image_patches = np.zeros((len(image_paths), patch_size, patch_size, 3))
    for i in range(no_of_images):
        x, y, w, h = sample_abnormal_ROIs[i]
        current_image = cv2.imread(image_paths[i])
        # current_image_1 = current_image.copy()
        # current_image_2 = current_image.copy()
        # current_image_3 = current_image.copy()
        # current_image_4 = current_image.copy()
        # new_image_1 = cv2.rectangle(current_image_1,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.imwrite('temp_old.jpg',new_image_1)
        # # current_image_2 = current_image
        # new_image_2 = cv2.rectangle(current_image_2,(x,y),(x+h,y+w),(0,255,0),3)
        # cv2.imwrite('temp_wh.jpg',new_image_2)
        # # current_image_3 = current_image
        # new_image_3 = cv2.rectangle(current_image_3,(y,x),(y+h,x+w),(0,255,0),3)
        # cv2.imwrite('temp_xywh.jpg',new_image_3)
        # new_image_4 = cv2.rectangle(current_image_4,(y,x),(y+w,x+h),(0,255,0),3)
        # cv2.imwrite('temp_xy.jpg',new_image_4)
        current_imgae = current_image.astype(np.float32)
        current_image = current_image / 255.0
        
        patch_x = np.NaN
        patch_y = np.NaN

        if w > patch_size:
            patch_x = random.choice(range(x, x+w-patch_size))
        elif w == patch_size:
            patch_x = x
        else:
            patch_x = random.choice(range(x+w-patch_size, x+patch_size-w))
            while (patch_x + patch_size >= current_image.shape[0] or patch_x < 0):
                patch_x = random.choice(range(x+w-patch_size, x))

        if h > patch_size:
            patch_y = random.choice(range(y, y+h-patch_size))
        elif h == patch_size:
            patch_y = y
        else:
            patch_y = random.choice(range(y+h-patch_size, y+patch_size-h))
            while(patch_y + patch_size >= current_image.shape[1] or patch_y < 0):
                patch_y = random.choice(range(y+h-patch_size, y))

        if patch_x == np.NaN or patch_y == np.NaN:
            print(image_paths[i])
            print(x, y, w, h)
            exit()

        current_patch = current_image[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size, :]
        
        try:
            image_patches[i,:,:,:] = current_patch
            # image_patches[i,:,:,:] = current_image
        except:
            print(image_paths[i])
            print(x, y, w, h)
            print(patch_x, patch_y)
            exit()

    return image_patches
        

def load_data_(image_paths):
    no_of_images = len(image_paths)
    images = []
    for i in range(no_of_images):
        # current_image = np.load(image_paths[i])
        current_image = cv2.imread(image_paths[i])
        # current_image = cv2.resize(current_file,(450,300))
        # current_image = current_image/255.
        images.append(current_image)
    images = np.asarray(images) # BBBAAAAADDDD idea in case of our problem - memory error will happen
    return images

def sample_images(normal_image_paths, abnormal_image_paths, abnormal_ROIs, nof_abnormal_image_samples, nof_normal_image_samples):
    # Sample classes from the dataset
    i = 0

    # Sample images from these classes until we have enough
    abnormal_image_indices = np.arange(len(abnormal_image_paths))
    np.random.shuffle(abnormal_image_indices)
    idx = abnormal_image_indices[:nof_abnormal_image_samples]
    image_paths = [abnormal_image_paths[j] for j in idx]
    sample_ROIs = [abnormal_ROIs[j] for j in idx]

    normal_image_indices = np.arange(len(normal_image_paths))
    np.random.shuffle(normal_image_indices)
    idx_ = normal_image_indices[:nof_normal_image_samples]
    image_paths.extend([normal_image_paths[j] for j in idx_])

    nof_sample_normal_ROIs = len(image_paths) - len(sample_ROIs)
    sample_normal_ROIs = [random.choice(sample_ROIs) for i in range(nof_sample_normal_ROIs)]
    sample_ROIs.extend(sample_normal_ROIs)

    num_per_class = [nof_abnormal_image_samples, nof_normal_image_samples]

    return image_paths, sample_ROIs, num_per_class

def sample_people(dataset, nof_classes_per_batch, nof_abnormal_image_samples):
    # 1000 abnormal images, 1000 normal images. 
    # nrof_images = 2 classes * 1000
    nrof_images = nof_classes_per_batch * nof_abnormal_image_samples

    # Sample classes from the dataset
    nrof_classes = len(dataset) # 2
    class_indices = np.arange(nrof_classes) # say [0, 1]
    np.random.shuffle(class_indices) #say [1, 0]

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, nof_abnormal_image_samples, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class # labels? why are they not returned?
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
    # len(image_paths) == 2000, num_per_class = [1000, 1000] 
    return image_paths, num_per_class

def combine_embeddings(x):
    anchor = tf.expand_dims(x[0],axis=1)
    positive = tf.expand_dims(x[1],axis=1)
    negative = tf.expand_dims(x[2],axis=1)

    return(tf.concat([anchor,positive,negative],axis=1))

# def triplet_loss(alpha):
#     def loss_func(y_true,y_pred):
#         pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[:,0,:], y_pred[:,1,:])), 1)
#         neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[:,0,:], y_pred[:,2,:])), 1)
        
#         basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
#         loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
#         return loss
#     return loss_func

def triplet_loss_func(alpha,y_pred):
    alpha = tf.squeeze(alpha,[1,2])
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[:,0,:], y_pred[:,1,:])), 1) # mse(anchor, positive)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[:,0,:], y_pred[:,2,:])), 1) # mse(anchor, negative)
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    # print(basic_loss)
    # exit()
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
  
    return loss

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_abnormal_ROIs(data_dir, box_csv_path):
    abnormal_ROIs = []
    with open(box_csv_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        count = 0
        for row in csvreader:
            if count > 0:
                x, y, w, h = [round(float(row[2])/2.5), round(float(row[3])/2.5), round(float(row[4])/2.5), round(float(row[5])/2.5)]
                abnormal_ROIs.append([x, y, w, h])

            count += 1
    
    return abnormal_ROIs

def get_abnormal_image_paths(data_dir, box_csv_path):
    abnormal_image_paths = []
    abnormal_ROIs = []
    with open(box_csv_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        count = 0
        for row in csvreader:
            if count > 0:
                for root, dirs, files in os.walk(data_dir, topdown=True):
                    if files is not None:
                        for each in files:
                            if each == row[0]:
                                abnormal_image_paths.append(os.path.join(root, each))
            count += 1

    with open("abnormal_image_paths.txt", 'w') as f:
        for s in abnormal_image_paths:
            f.write(str(s) + '\n')

    return abnormal_image_paths

def get_normal_image_paths(data_dir, data_csv_path):
    normal_image_paths = []
    with open(data_csv_path, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        count = 0
        for row in csvreader:
            if count > 0:
                if row[1] == 'No Finding':
                    stop_walk_flag = 0
                    for root, dirs, files in os.walk(data_dir, topdown=True):
                        if files is not None:
                            for each in files:
                                if each == row[0]:
                                    normal_image_paths.append(os.path.join(root, each))
                                    stop_walk_flag = 1
                                    break
                        if stop_walk_flag == 1:
                            break
            count += 1

    with open("normal_image_paths.txt", 'w') as f:
        for s in normal_image_paths:
            f.write(str(s) + '\n')
    print(count)
    print(len(normal_image_paths))
    return normal_image_paths

def get_dataset(data_dir, data_csv_path, box_csv_path, has_class_directories=True):
    dataset = []
    data_dir = os.path.expanduser(data_dir)

    classes = ['normal', 'abnormal']
    # normal_image_paths = get_normal_image_paths(data_dir, data_csv_path)
    with open("normal_image_paths.txt", 'r') as f:
        normal_image_paths = [line.rstrip('\n') for line in f]
    print('num of normal_image_paths', len(normal_image_paths))

    # abnormal_image_paths = get_abnormal_image_paths(data_dir, box_csv_path)
    with open("abnormal_image_paths.txt", 'r') as f:
        abnormal_image_paths = [line.rstrip('\n') for line in f]
    print('num of abnormal_image_paths)', len(abnormal_image_paths))

    abnormal_ROIs = get_abnormal_ROIs(data_dir, box_csv_path)
    abnormal_ROIs = np.asarray(abnormal_ROIs)
    print('shape of abnormal_ROIs', abnormal_ROIs.shape)

    return normal_image_paths, abnormal_image_paths, abnormal_ROIs

def get_dataset_(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)

    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]

    classes.sort()
    classes = ['normal', 'abnormal']
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        # here will have binary class - normal (pick from data.csv sheet) and abnormal (taken only from the box.csv sheet)
        classdir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(classdir) # Contains paths of all images in the class
        dataset.append(ImageClass(class_name, image_paths))
        #dataset will contain list of ImageClass objects.
        #print(dataset[0]) calls the __str__ function
        #print(len(dataset[0])) calls the __len__ function
        
    return dataset

def get_image_paths(classdir):
    image_paths = []
    if os.path.isdir(classdir):
        images = os.listdir(classdir)
        image_paths = [os.path.join(classdir,img) for img in images]
    return image_paths













