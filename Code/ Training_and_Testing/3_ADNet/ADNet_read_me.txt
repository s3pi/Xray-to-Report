RedCRFENet.py:
Used for extracting image features from small patches of Xrays.
Maxpooling2D layers are reduced to 3 from 4 in CRFENet.
Batchnormalization layer is removed.

train_classifier_RedCRFENet_with_NIH_data.py
Train a classifier model : using RedCRFENet followed by GAP layer and Dense(15) layer.
Data used is NIH data with ROI to classify between abnormal and normal patches.
RedCRFENet produces 10x10x512 feature in the end. Save the model weights (say classifier_model_weights) from beginning uptill this layery

distance_net.py:
Loads data in patches, selects triplets.
Used by train_classifier_RedCRFENet_with_NIH_data.py and EG_with_triplet_loss.py

model_siamese.py:
Creates model by loading classifier_model_weights onto RedCRFENet.py followed by GAP layer adn Dense(bottle_neck_layer_size = 512) to get 1x512 for every patch.
Used by EG_with_triplet_loss.py

EG_with_triplet_loss.py:
Uses model_siamese.RedCRFENet to train with triplet loss such that normal and abnormal patches are as far as possible in the feature space. 
Trained on NIH data.




