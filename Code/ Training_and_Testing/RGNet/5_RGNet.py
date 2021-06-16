import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
import os
import numpy as np
import math
from sklearn.utils import shuffle
import nltk
from sys import exit

def write_metric_files(files, values):
    for i in range(len(files)):
        files[i].write(str(values[i])+ '\n')

def loss_function(real,pred):
    real_sparse = tf.argmax(real, axis=-1)
    mask = tf.math.logical_not(tf.math.equal(real_sparse, 0))
    
    
    
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    loss = tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    return loss

def metric_function(real, pred):
    pred = tf.argmax(pred, 2)
    real = tf.cast(tf.argmax(real, 2), pred.dtype)
    
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=pred.dtype)
    
    acc_ = tf.cast(tf.math.equal(real,pred), pred.dtype)
    
    acc_ *= mask

    acc = tf.reduce_sum(acc_)/tf.reduce_sum(mask)
    # acc = tf.reduce_mean(acc_)
    return acc

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = Wide_MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        # print(attn_output.shape)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.tb_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        # self.conv1 = tf.keras.layers.Conv2D(1,1,activation='relu', padding = 'valid')
    
    def call(self, x, training):

        for i in range(self.num_layers):
          x = self.tb_layers[i](x, training)
        
        # x = tf.expand_dims(x, 1)   # (batch_size, 1, seq_len, d_model)
        # x = tf.transpose(x, perm=[0, 1, 3, 2])  # (batch_size, 1, d_model, seq_length)
        # x = self.conv1(x)  # (batch_size, 1, d_model, 1)
        # x = tf.squeeze(x,[1,3])  # (batch_size, d_model)

        return x

class EncoderModel(tf.keras.Model):
    def __init__(self, num_layers, d_model_feat, num_heads_feat, dff_feat, rate=0.1):
        super(EncoderModel, self).__init__()
        self.feature_transformer_layer = Encoder(num_layers, d_model_feat, num_heads_feat, dff_feat, rate)
        # self.spatial_transformer_layer = Encoder(num_layers, d_model_spat, num_heads_spat, dff_spat, rate)
        self.fc1 = tf.keras.layers.Dense(d_model_feat,activation='tanh')
        # self.fc1 = tf.keras.layers.Dense(num_classes, activation='sigmoid')
    
    # def call(self, x, saved_features_from_TagModel, training):
    def call(self, new_x, training):
        x = new_x
        # saved_features_from_TagModel = new_x[1]

        x1 = tf.reshape(x,[-1,x.shape[1]*x.shape[2],x.shape[3]])   # bs , d_model_feat , 1024
        x2 = tf.transpose(x1,[0, 2, 1])       # bs, 1024, d_model_feat
        x2 = self.fc1(x2)   # bs, 1024, d_model_feat

        # y1 = self.spatial_transformer_layer(x1, training=training)  # bs, 1024
        y2 = self.feature_transformer_layer(x2, training=training)  # bs , 1024, d_model_feat

        # z = tf.concat([y1,y2],axis=1)   # bs, 1105
        # enc_model_output = tf.concat([y2, saved_features_from_TagModel],axis=1)
        #op = self.fc1(z)   # bs, 210 (used for tag classification)
        
        return y2


class DecoderLayer_imp(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer_imp, self).__init__()

    self.mha1 = Narrow_MultiHeadAttention(d_model, num_heads)
    self.mha2 = Narrow_MultiHeadAttention(d_model, num_heads)
    self.mha3 = Narrow_MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    self.dropout4 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_model_output, dec_fin_out, training, look_ahead_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    ##### Self attention btw imp ####
    attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask)
    # attn1, attn_weights_block1 = self.mha1(x, x, x)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    # out1 = attn1
    
    ##### Cross attention btw encoder output and ou12 ####    
    attn2, attn_weights_block2 = self.mha2(
        enc_model_output, enc_model_output, out1, mask=None)  # (batch_size, target_seq_len, d_model)

    # attn2, attn_weights_block2 = self.mha2(
        # out1, out1, out1)
    # attn2, attn_weights_block2 = self.mha2(
    #     out1, out1, out1, mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)

    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ##### Cross attention btw decoder_fin output and out2 ####
    attn3, attn_weights_block3 = self.mha3(
        dec_fin_out, dec_fin_out, out2, mask=None)  # (batch_size, target_seq_len, d_model)
    attn3 = self.dropout3(attn3, training=training)
    out3 = self.layernorm3(attn2 + out2)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out3)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout4(ffn_output, training=training)
    out4 = self.layernorm4(ffn_output + out3)  # (batch_size, target_seq_len, d_model)
    
    return out4, attn_weights_block1, attn_weights_block1

class DecoderModel_imp(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(DecoderModel_imp, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    # self.pos_encoding2 = positional_encoding(maximum_position_encoding, 1186)
    
    self.dec_layers = [DecoderLayer_imp(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    # self.fc1 = tf.keras.layers.Dense(256,activation='tanh')
    self.final_layer = tf.keras.layers.Dense(target_vocab_size,activation = 'softmax', kernel_initializer = tf.keras.initializers.lecun_normal())

  # def call(self, x, enc_model_output, look_ahead_mask, training):
  def call(self, new_x, training):
    x = new_x[0]
    enc_model_output = new_x[1]
    dec_fin_out = new_x[2]

    seq_len = tf.shape(x)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)
    attention_weights = {}

    x += self.pos_encoding[:, :seq_len, :]
      # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
    
    x = self.dropout(x, training=training)

    # enc_feature_dim = enc_model_output.shape[-1]
    # enc_model_output = tf.expand_dims(enc_model_output,1)
    # print(seq_len)
    # exit()
    # enc_model_output = tf.keras.backend.repeat_elements(enc_model_output, 261,1)
    # print(enc_model_output)
    # exit()
    # enc_model_output+=self.pos_encoding2[:,:seq_len,:]
    # for i in range(self.num_layers):
    #   x, block1, block2 = self.dec_layers[i](x, enc_model_output, training, look_ahead_mask)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_model_output, dec_fin_out, training, look_ahead_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)

    final_output = self.final_layer(x)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights

class DecoderLayer_fin(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer_fin, self).__init__()

    self.mha1 = Narrow_MultiHeadAttention(d_model, num_heads)
    self.mha2 = Narrow_MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output,training, 
           look_ahead_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, mask=look_ahead_mask)
    # attn1, attn_weights_block1 = self.mha1(x, x, x)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    # out1 = attn1
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, mask=None)  # (batch_size, target_seq_len, d_model)
    # attn2, attn_weights_block2 = self.mha2(
        # out1, out1, out1)
    # attn2, attn_weights_block2 = self.mha2(
    #     out1, out1, out1, mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)

    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    # out2 = attn2
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block1

class DecoderModel_fin(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(DecoderModel_fin, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    # self.pos_encoding2 = positional_encoding(maximum_position_encoding, 1186)
    
    self.dec_layers = [DecoderLayer_fin(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    # self.fc1 = tf.keras.layers.Dense(256,activation='tanh')
    self.final_layer = tf.keras.layers.Dense(target_vocab_size,activation = 'softmax', kernel_initializer = tf.keras.initializers.lecun_normal())

  # def call(self, x, enc_model_output, look_ahead_mask, training):
  def call(self, new_x, training):
    x = new_x[0]
    enc_model_output = new_x[1]
    # print(tf.shape(x)[1])
    # exit()
    seq_len = tf.shape(x)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)
    attention_weights = {}

    x += self.pos_encoding[:, :seq_len, :]
      # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
    
    x = self.dropout(x, training=training)

    # enc_feature_dim = enc_model_output.shape[-1]
    # enc_model_output = tf.expand_dims(enc_model_output,1)
    # print(seq_len)
    # exit()
    # enc_model_output = tf.keras.backend.repeat_elements(enc_model_output, 261,1)
    # print(enc_model_output)
    # exit()
    # enc_model_output+=self.pos_encoding2[:,:seq_len,:]
    # for i in range(self.num_layers):
    #   x, block1, block2 = self.dec_layers[i](x, enc_model_output, training, look_ahead_mask)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_model_output, training, look_ahead_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)

    final_output = self.final_layer(x)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights


def make_model():
    enc_model = EncoderModel(num_layers_encoder, d_model_feat, num_heads_feat, dff_feat, rate)
    dec_fin_model = DecoderModel_fin(num_layers_decoder, d_model_decoder, num_heads_decoder, dff_decoder, target_vocab_size, pe_target, rate)
    dec_imp_model = DecoderModel_imp(num_layers_decoder, d_model_decoder, num_heads_decoder, dff_decoder, target_vocab_size, pe_target, rate)
    embedding_model = tf.keras.Sequential(tf.keras.layers.Embedding(target_vocab_size, d_model_decoder))
    
    inp_to_enc_model = tf.keras.layers.Input((9,9,1024))
    # feat_from_TagModel = tf.keras.layers.Input((1105,))

    enc_op = enc_model(inp_to_enc_model)

    inp_to_dec_fin_model = tf.keras.layers.Input((None,))
    x = embedding_model(inp_to_dec_fin_model)
    fin_op, fin_att = dec_fin_model([x, enc_op])
    fin_model = tf.keras.Model([inp_to_enc_model,inp_to_dec_fin_model], fin_op)
    fin_model_vis = tf.keras.Model([inp_to_enc_model,inp_to_dec_fin_model], fin_att)
    # fin_model.summary()
    fin_model.compile(loss=loss_function,optimizer = tf.keras.optimizers.Adam(1e-4, clipnorm = 1.), metrics=[metric_function])

    inp_to_dec_imp_model = tf.keras.layers.Input((None,))
    y = embedding_model(inp_to_dec_imp_model)
    # print(fin_op.shape)
    # exit()
    imp_op, imp_att = dec_imp_model([y, enc_op, fin_op])
    imp_model = tf.keras.Model([inp_to_enc_model, inp_to_dec_fin_model, inp_to_dec_imp_model], imp_op)
    imp_model_vis = tf.keras.Model([inp_to_enc_model, inp_to_dec_fin_model, inp_to_dec_imp_model], imp_att)
    imp_model.summary()
    exit()
    imp_model.compile(loss=loss_function,optimizer = tf.keras.optimizers.Adam(1e-4, clipnorm = 1.), metrics=[metric_function])
    
    return fin_model, imp_model, embedding_model, dec_fin_model, dec_imp_model, enc_model

def load_inp_to_dec_fin(path):
    with open(path, 'r') as f:
        a = f.readlines()
        num_of_files = len(a)
        data = np.zeros((num_of_files,189, 1))
        labels = np.zeros((num_of_files,189, 1))
        for i in range(num_of_files):
            file = a[i].split()
            file_name = file[0]
            labels_file_path = os.path.join(img_fin_path, file_name)
            current_data = np.load(labels_file_path)
            current_data = np.expand_dims(current_data, 1)
            data[i,:,:] = current_data[:-1,:]
            labels[i] = current_data[1:,:]

    return data, labels

def load_inp_to_dec_imp(path):
    with open(path, 'r') as f:
        a = f.readlines()
        num_of_files = len(a)
        data = np.zeros((num_of_files,139, 1))
        labels = np.zeros((num_of_files,139, 1))
        for i in range(num_of_files):
            file = a[i].split()
            file_name = file[0]
            labels_file_path = os.path.join(img_imp_path, file_name)
            current_data = np.load(labels_file_path)
            current_data = np.expand_dims(current_data, 1)
            data[i,:,:] = current_data[:-1,:]
            labels[i] = current_data[1:,:]

    return data, labels

def load_inp_to_enc_model(path):
    with open(path, 'r') as f:
        a = f.readlines()
        data = np.zeros((len(a),9,9,1024))
        for i in range(len(a)):
            file = a[i].split()
            file_name = file[0]
            file_path = os.path.join(IU_features_path, file_name)
            data[i] = np.load(file_path)

    return data


def load_saved_features_from_TagModel(path):
    with open(path, 'r') as f:
        a = f.readlines()
        data = np.zeros((len(a),1105))
        for i in range(len(a)):
            file = a[i].split()
            file_name = file[0]
            file_path = os.path.join(TagModel_features_path, file_name)
            data[i] = np.load(file_path)

    return data

def load_data(mode):
    if mode is 'train':
        
        train_filenames_txt_path = os.path.join(IU_6461_img_level_details_path, 'train_filenames.txt')
        val_filenames_txt_path = os.path.join(IU_6461_img_level_details_path, 'val_filenames.txt')
        
        print("1")
        inp_to_enc_model_train_data = load_inp_to_enc_model(train_filenames_txt_path) #inp_to_enc_model
        inp_to_enc_model_val_data = load_inp_to_enc_model(val_filenames_txt_path) #inp_to_enc_model
        
        # saved_features_from_TagModel_train_data = load_saved_features_from_TagModel(train_filenames_txt_path) #feat_from_TagModel
        # saved_features_from_TagModel_val_data = load_saved_features_from_TagModel(val_filenames_txt_path) #feat_from_TagModel
        print("2")
        inp_to_dec_fin_model_train_data, inp_to_dec_fin_model_train_labels = load_inp_to_dec_fin(train_filenames_txt_path)
        inp_to_dec_fin_model_val_data, inp_to_dec_fin_model_val_labels = load_inp_to_dec_fin(val_filenames_txt_path)
        print("3")
        inp_to_dec_imp_model_train_data, inp_to_dec_imp_model_train_labels = load_inp_to_dec_imp(train_filenames_txt_path)
        inp_to_dec_imp_model_val_data, inp_to_dec_imp_model_val_labels = load_inp_to_dec_imp(val_filenames_txt_path)

        # saved_features_from_TagModel_train_data, saved_features_from_TagModel_val_data, \
        return inp_to_enc_model_train_data, inp_to_enc_model_val_data, \
        inp_to_dec_fin_model_train_data, inp_to_dec_fin_model_train_labels, \
        inp_to_dec_fin_model_val_data, inp_to_dec_fin_model_val_labels, \
        inp_to_dec_imp_model_train_data, inp_to_dec_imp_model_train_labels, \
        inp_to_dec_imp_model_val_data, inp_to_dec_imp_model_val_labels

    if mode is 'test':
        test_filenames_txt_path = os.path.join(IU_6461_img_level_details_path, 'test_filenames.txt')
        inp_to_enc_model_test_data = load_inp_to_enc_model(test_filenames_txt_path) #inp_to_enc_model
        saved_features_from_TagModel_test_data = load_saved_features_from_TagModel(test_filenames_txt_path) #feat_from_TagModel
        inp_to_dec_model_test_data, inp_to_dec_model_test_label = load_inp_to_dec_fin(test_filenames_txt_path)

        return inp_to_enc_model_test_data ,inp_to_dec_model_test_data, inp_to_dec_model_test_label

def open_metric_files():
    per_batch_train_metrics_file = open(result_files_path + '/training_per_batch_metrics' + '.txt', 'a')
    per_epoch_train_metrics_file = open(result_files_path + '/training_per_epoch_metrics' + '.txt', 'a')
    per_epoch_val_metrics_file = open(result_files_path + '/test_metrics' + '.txt', 'a')

    return per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file

def to_categorical(array):
    array = tf.keras.utils.to_categorical(array, num_classes=target_vocab_size)

    return array

def train():
    fin_model, imp_model, embedding_model, dec_fin_model, dec_imp_model, enc_model = make_model()
    fin_model.summary()
    exit()
    # saved_features_from_TagModel_train_data, saved_features_from_TagModel_val_data, \
    inp_to_enc_model_train_data, inp_to_enc_model_val_data, \
    inp_to_dec_fin_model_train_data, inp_to_dec_fin_model_train_labels, \
    inp_to_dec_fin_model_val_data, inp_to_dec_fin_model_val_labels, \
    inp_to_dec_imp_model_train_data, inp_to_dec_imp_model_train_labels, \
    inp_to_dec_imp_model_val_data, inp_to_dec_imp_model_val_labels = load_data('train')
    
    inp_to_dec_fin_model_train_labels = to_categorical(inp_to_dec_fin_model_train_labels)
    inp_to_dec_fin_model_val_labels = to_categorical(inp_to_dec_fin_model_val_labels)
    inp_to_dec_imp_model_train_labels = to_categorical(inp_to_dec_imp_model_train_labels)
    inp_to_dec_imp_model_val_labels = to_categorical(inp_to_dec_imp_model_val_labels)

    save_fin_model_list=[]
    save_imp_model_list=[]
    test_e = 0
    # prev_epoch = -1
    min_fin_loss = 100.0
    min_imp_loss = 100.0
    num_of_batches = int(math.ceil(inp_to_dec_fin_model_train_data.shape[0]/batch_size))
    per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file = open_metric_files()
    for e in range(num_epochs):
        per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file = open_metric_files()
        
        inp_to_enc_model_train_data, \
        inp_to_dec_fin_model_train_data, inp_to_dec_fin_model_train_labels, \
        inp_to_dec_imp_model_train_data, inp_to_dec_imp_model_train_labels = \
        shuffle(inp_to_enc_model_train_data, \
        inp_to_dec_fin_model_train_data, inp_to_dec_fin_model_train_labels, \
        inp_to_dec_imp_model_train_data, inp_to_dec_imp_model_train_labels, random_state = 2)

        per_epoch_train_fin_loss, per_epoch_train_fin_acc = 0.0, 0.0
        per_epoch_train_imp_loss, per_epoch_train_imp_acc = 0.0, 0.0

        for batch_num in range(num_of_batches):
        # for batch_num in range(2):
            batch_X_train_inp_to_enc_model = np.zeros((batch_size, 9, 9, 1024))
            # batch_X_train_saved_features_from_TagModel = np.zeros((batch_size, 1105))
            batch_X_train_inp_to_dec_fin_model = np.zeros((batch_size, 189))
            batch_y_train_inp_to_dec_fin_model = np.zeros((batch_size, 189, target_vocab_size))
            batch_X_train_inp_to_dec_imp_model = np.zeros((batch_size, 139))
            batch_y_train_inp_to_dec_imp_model = np.zeros((batch_size, 139, target_vocab_size))
            b = 0
            for j in range(batch_num*batch_size, min((batch_num+1)*batch_size, inp_to_dec_fin_model_train_data.shape[0])):
                batch_X_train_inp_to_enc_model[b, :, :, :] = inp_to_enc_model_train_data[j,:,:,:]
                # batch_X_train_saved_features_from_TagModel[b,:] = saved_features_from_TagModel_train_data[j,:]
                batch_X_train_inp_to_dec_fin_model[b,:] = inp_to_dec_fin_model_train_data[j,:,0]       
                batch_y_train_inp_to_dec_fin_model[b, :, :] = inp_to_dec_fin_model_train_labels[j,:,:]
                batch_X_train_inp_to_dec_imp_model[b,:] = inp_to_dec_imp_model_train_data[j,:,0]       
                batch_y_train_inp_to_dec_imp_model[b, :, :] = inp_to_dec_imp_model_train_labels[j,:,:]
                b += 1
            ####### training fin model on batch ##########
            batch_fin_X_train = [batch_X_train_inp_to_enc_model,batch_X_train_inp_to_dec_fin_model]
            per_batch_fin_train_loss, per_batch_fin_train_acc = fin_model.train_on_batch(batch_fin_X_train, batch_y_train_inp_to_dec_fin_model)
          
            print('fin model \n epoch_num: %d, batch_num: %d, loss: %f, class_wise_accuracy: %s\n' % (e, batch_num, per_batch_fin_train_loss, per_batch_fin_train_acc))
       
            write_metric_files([per_batch_train_metrics_file], [['fin model', e, batch_num, per_batch_fin_train_loss, per_batch_fin_train_acc]])

            per_epoch_train_fin_loss += per_batch_fin_train_loss
            per_epoch_train_fin_acc += per_batch_fin_train_acc

            ####### training fin model on batch ##########
            batch_imp_X_train = [batch_X_train_inp_to_enc_model,batch_X_train_inp_to_dec_fin_model, batch_X_train_inp_to_dec_imp_model]
            per_batch_imp_train_loss, per_batch_imp_train_acc = imp_model.train_on_batch(batch_imp_X_train, batch_y_train_inp_to_dec_imp_model)
          
            print('imp model \n epoch_num: %d, batch_num: %d, loss: %f, class_wise_accuracy: %s\n' % (e, batch_num, per_batch_imp_train_loss, per_batch_imp_train_acc))
       
            write_metric_files([per_batch_train_metrics_file], [['imp model', e, batch_num, per_batch_imp_train_loss, per_batch_imp_train_acc]])

            per_epoch_train_imp_loss += per_batch_imp_train_loss
            per_epoch_train_imp_acc += per_batch_imp_train_acc

        print('---------------------------------------------------------------------\n')
        per_epoch_fin_val_loss, per_epoch_fin_val_acc = 0.0, 0.0
        per_epoch_imp_val_loss, per_epoch_imp_val_acc = 0.0, 0.0
        for i in range(inp_to_dec_fin_model_val_data.shape[0]):
            ######## predict for imp model on batch #########
            # temp = tf.expand_dims(inp_to_enc_model_val_data[i,:,:,:], 0)
            fin_val_data = [inp_to_enc_model_val_data[i:i+1,:,:,:],inp_to_dec_fin_model_val_data[i:i+1,:,0]]
            curr_fin_val_loss, curr_fin_val_acc = fin_model.test_on_batch(fin_val_data, inp_to_dec_fin_model_val_labels[i:i+1,:,:])
            per_epoch_fin_val_loss += curr_fin_val_loss
            per_epoch_fin_val_acc += curr_fin_val_acc

            ######## predict for imp model on batch #########
            imp_val_data = [inp_to_enc_model_val_data[i:i+1,:,:,:],inp_to_dec_fin_model_val_data[i:i+1,:,0], inp_to_dec_imp_model_val_data[i:i+1,:,0]]
            curr_imp_val_loss, curr_imp_val_acc = imp_model.test_on_batch(imp_val_data, inp_to_dec_imp_model_val_labels[i:i+1,:,:])
            per_epoch_imp_val_loss += curr_imp_val_loss
            per_epoch_imp_val_acc += curr_imp_val_acc
        
        ######## write fin model details #########
        test_e+=1
        per_epoch_fin_val_loss /= inp_to_dec_fin_model_val_data.shape[0]
        per_epoch_fin_val_acc /= inp_to_dec_fin_model_val_data.shape[0]
        print('fin model \n test_num: %d, loss: %f, \nclass_wise_accuracy: %s\n' % (test_e, per_epoch_fin_val_loss, per_epoch_fin_val_acc))

        ######## write imp model details #########
        per_epoch_imp_val_loss /= inp_to_dec_imp_model_val_data.shape[0]
        per_epoch_imp_val_acc /= inp_to_dec_imp_model_val_data.shape[0]
        print('imp model \n test_num: %d, loss: %f, \nclass_wise_accuracy: %s\n' % (test_e, per_epoch_imp_val_loss, per_epoch_imp_val_acc))     

        ######## save fin model weights and write fin model val details #########
        if per_epoch_fin_val_loss < min_fin_loss:
            save_fin_model_list.append(test_e)
            fin_model.save_weights(fin_model_weights_path + '/fin_model_' + str(test_e)+'.h5')
            embedding_model.save_weights(fin_model_weights_path + '/emb_model_' + str(test_e)+'.h5')
            dec_fin_model.save_weights(fin_model_weights_path + '/dec_fin_model_' + str(test_e)+'.h5')
            enc_model.save_weights(fin_model_weights_path + '/enc_model_' + str(test_e)+'.h5')
            if len(save_fin_model_list) > num_of_saved_models:
                del_model_count = save_fin_model_list.pop(0)
                os.remove(fin_model_weights_path + '/fin_model_' + str(del_model_count)+'.h5')
                os.remove(fin_model_weights_path + '/emb_model_' + str(del_model_count)+'.h5')
                os.remove(fin_model_weights_path + '/dec_fin_model_' + str(del_model_count)+'.h5')
                os.remove(fin_model_weights_path + '/enc_model_' + str(del_model_count)+'.h5')
            # prev_epoch = 0
            print('Fin model loss improved from %f to %f. Saving Weights\n'%(min_fin_loss, per_epoch_fin_val_loss))
            min_fin_loss = per_epoch_fin_val_loss
        else:
            print('Best fin loss: %f\n'%(min_fin_loss))
        print('----------------------------------------------------------------------\n')
        write_metric_files([per_epoch_val_metrics_file], [['fin model', min_fin_loss, test_e, per_epoch_fin_val_loss, per_epoch_fin_val_acc]])

        per_epoch_train_fin_loss = per_epoch_train_fin_loss / num_of_batches
        per_epoch_train_fin_acc = per_epoch_train_fin_acc / num_of_batches
        write_metric_files([per_epoch_train_metrics_file], [['fin model', e, per_epoch_train_fin_loss, per_epoch_train_fin_acc]])

        ######## save imp model weights and write imp model val details #########
        if per_epoch_imp_val_loss < min_imp_loss:
            save_imp_model_list.append(test_e)
            imp_model.save_weights(imp_model_weights_path + '/imp_model_' + str(test_e)+'.h5')
            embedding_model.save_weights(imp_model_weights_path + '/emb_model_' + str(test_e)+'.h5')
            dec_fin_model.save_weights(imp_model_weights_path + '/dec_fin_model_' + str(test_e)+'.h5')
            dec_imp_model.save_weights(imp_model_weights_path + '/dec_imp_model_' + str(test_e)+'.h5')
            enc_model.save_weights(imp_model_weights_path + '/enc_model_' + str(test_e)+'.h5')
            if len(save_imp_model_list) > num_of_saved_models:
                del_model_count = save_imp_model_list.pop(0)
                os.remove(imp_model_weights_path + '/imp_model_' + str(del_model_count)+'.h5')
                os.remove(imp_model_weights_path + '/emb_model_' + str(del_model_count)+'.h5')
                os.remove(imp_model_weights_path + '/dec_fin_model_' + str(del_model_count)+'.h5')
                os.remove(imp_model_weights_path + '/dec_imp_model_' + str(del_model_count)+'.h5')
                os.remove(imp_model_weights_path + '/enc_model_' + str(del_model_count)+'.h5')
            # prev_epoch = 0
            print('imp model loss improved from %f to %f. Saving Weights\n'%(min_imp_loss, per_epoch_imp_val_loss))
            min_imp_loss = per_epoch_imp_val_loss
        else:
            print('Best imp loss: %f\n'%(min_imp_loss))
        print('----------------------------------------------------------------------\n')
        write_metric_files([per_epoch_val_metrics_file], [['imp model', min_imp_loss, test_e, per_epoch_imp_val_loss, per_epoch_imp_val_acc]])

        per_epoch_train_imp_loss = per_epoch_train_imp_loss / num_of_batches
        per_epoch_train_imp_acc = per_epoch_train_imp_acc / num_of_batches
        write_metric_files([per_epoch_train_metrics_file], [['imp model', e, per_epoch_train_imp_loss, per_epoch_train_imp_acc]])

def test_best_fin():
    final_model,_,_,_ = make_model()
    final_model.load_weights(fin_model_weights_path+'/final_model_70.h5')
    
    inp_to_enc_model_test_data, inp_to_dec_model_test_data, inp_to_dec_model_test_label = load_data('test')
    # print('YAAAA')
    # exit()
    test_acc1 = 0.0
    test_acc2 = 0.0
    test_acc3 = 0.0
    test_acc4 = 0.0
    test_acc5 = 0.0
    count=0
    for i in range(inp_to_dec_model_test_data.shape[0]):
        print('Testing '+str(i+1)+' sample out of '+str(inp_to_dec_model_test_data.shape[0]))
        ref2=[]
        # print(inp_to_dec_model_test_label.shape)
        for j in range(261):
            #to remove '.', 'end character' and 'pad'
            if (inp_to_dec_model_test_label[i,j,0] != 1173) and (inp_to_dec_model_test_label[i,j,0] != 1) and (inp_to_dec_model_test_label[i,j,0] != 0):
                ref2.append(str(int(inp_to_dec_model_test_label[i,j,0])))
        # print(ref2)
        if len(ref2)>=5:
            index = 2
            text_input_list = []
            while (index != 1) and (len(text_input_list)<262):
                text_input_list.append(index)
                inp_text = np.asarray(text_input_list)
                inp_text = np.expand_dims(inp_text,0)
                out = final_model.predict([inp_to_enc_model_test_data[i:i+1,:,:,:],inp_text])
                out = np.squeeze(out,0)
                out = np.argmax(out,1)
                index = out[-1]
            # print(text_input_list)
            ref=[]
            for j in range(len(text_input_list)):
                if (text_input_list[j]!=1173) and (text_input_list[j]!=2):
                    ref.append(str(text_input_list[j]))
            # print(ref)
            if len(ref)>=5:
                score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1,))
                test_acc1+=score
                score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (0.5,0.5))
                test_acc2+=score
                score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1./3.,1./3.,1./3.))
                test_acc3+=score
                score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (0.25,0.25,0.25,0.25))
                test_acc4+=score
                score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1./5.,1./5.,1./5.,1./5.,1./5.))
                test_acc5+=score


            else:
                test_acc5+=0
                if len(ref)==4:
                    score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1,))
                    test_acc1+=score
                    score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (0.5,0.5))
                    test_acc2+=score
                    score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1./3.,1./3.,1./3.))
                    test_acc3+=score
                    score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (0.25,0.25,0.25,0.25))
                    test_acc4+=score
                else:
                    test_acc4+=0
                    if len(ref)==3:
                        score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1,))
                        test_acc1+=score
                        score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (0.5,0.5))
                        test_acc2+=score
                        score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1./3.,1./3.,1./3.))
                        test_acc3+=score
                    else:
                        test_acc3+=0.0
                        if len(ref)==2:
                            score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1,))
                            test_acc1+=score
                            score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (0.5,0.5))
                            test_acc2+=score
                        else:
                            test_acc2+=0.0
                            if len(ref)==1:
                                score = nltk.translate.bleu_score.sentence_bleu([ref2], ref, weights = (1,))
                                test_acc1+=score
                            else:
                                test_acc1+=0.0
            count+=1
    test_acc1/=count
    test_acc2/=count
    test_acc3/=count
    test_acc4/=count
    test_acc5/=count
    print('Bleu 1: '+str(test_acc1)+'\nBleu 2: '+str(test_acc2)+'\nBleu3: '+str(test_acc3)+'\nBleu 4: '+str(test_acc4)+'\nBleu 5: '+str(test_acc5))

        
#################################### Global Arguments #######################################
loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, reduction='none')
num_layers_decoder = 6
d_model_decoder = 512
num_heads_decoder = 8
dff_decoder = 512
pe_target=1000
rate=0.3
num_layers_encoder = 2
d_model_feat=128
d_model_spat=1024
num_heads_feat=2 
num_heads_spat=1
dff_feat=128
dff_spat=1024
target_vocab_size=1000
################# Preliminary Code Test ##########################
batch_size = 32
num_epochs = 2000
num_of_saved_models = 10
#############################################################
# server_path = "/mnt/Data2/Preethi/XRay_Report_Generation"
server_path = "/home/ada/Preethi/XRay_Report_Generation"
#############################################################
# IU_7430_data_details_path = server_path + "/Data/IU_7430_data_details"
IU_6461_img_level_details_path = "/home/ada/Preethi/XRay_Report_Generation/Data/IU_6461_img_level_details"
IU_features_path = server_path + "/Data/IU_features"
TagModel_features_path = server_path + "/Data/tag_features"
img_fin_path = server_path + "/Data/text_data_img_findings"
img_imp_path = server_path + "/Data/text_data_img_impressions"
result_files_path = server_path + "/Code/6_report_generation/Enc_with_2_Decs_for_Find_and_Imp/Results"
fin_model_weights_path = server_path + "/Code/6_report_generation/Enc_with_2_Decs_for_Find_and_Imp/Model_Weights_fin"
imp_model_weights_path = server_path + "/Code/6_report_generation/Enc_with_2_Decs_for_Find_and_Imp/Model_Weights_imp"
#################################### Global Arguments #######################################

train()
# test()

