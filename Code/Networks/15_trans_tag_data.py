import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, LayerNormalization
import os
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask = None):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2


class Encoder(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, ip, training):

    x = ip[0]
    mask = ip[1]
    seq_len = tf.shape(x)[1]
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.mha3 = MultiHeadAttention(d_model, num_heads)
    
    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    self.dropout4 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, data_enc_op,training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    attn3, attn_weights_block3 = self.mha3(
        data_enc_op, data_enc_op, out1, mask=None)  # (batch_size, target_seq_len, d_model)
    attn3 = self.dropout3(attn3, training=training)
    out3 = self.layernorm2(attn3 + out2)

    ffn_output = self.ffn(out3)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout4(ffn_output, training=training)
    out4 = self.layernorm4(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out4, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    self.dense = tf.keras.layers.Dense(target_vocab_size,activation='softmax')
    
  def call(self, ip, training):
    x = ip[0]
    enc_output = ip[1]
    data_enc_op = ip[2]
    padding_mask = ip[3]
    seq_len = tf.shape(x)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, data_enc_op, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    x = self.dense(x)
    return x


class Data_Encoder(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Data_Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.fc1 = tf.keras.layers.Dense(d_model, activation = 'tanh')
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, ip, training):

    x = ip   # bs, 9 , 9 , 1024
    x = tf.reshape(x,[-1,x.shape[1]*x.shape[2],x.shape[3]])   # bs , 81 , 1024
    x = tf.transpose(x,[0, 2, 1])       # bs, 1024, 81
    x = self.fc1(x)
    

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training)
    
    return x  # (batch_size, input_seq_len, d_model)



def make_model():
    data_enc_model = Data_Encoder(data_enc_num_layers, data_enc_d_model, data_enc_num_heads, data_enc_dff, data_enc_rate)
    enc_model = Encoder(enc_num_layers, enc_d_model, enc_num_heads, enc_dff, number_of_tags, enc_rate )
    dec_model = Decoder(dec_num_layers, dec_d_model, dec_num_heads, dec_dff, target_vocab_size, maximum_position_encoding, dec_rate)

    ip3 = Input(shape=(10,10,512))
    data_enc_op = data_enc_model(ip3)
    ip1 = Input(shape=(None,))
    padding_mask = create_padding_mask(ip1)
    ip2 = Input(shape=(None,))
    enc_op = enc_model([ip1,padding_mask])
    # print(enc_op)
    # exit()
    op = dec_model([ip2,enc_op,data_enc_op,padding_mask])
    model = Model([ip1,ip3,ip2],op)
    model.summary()
    # exit()
    model.compile(loss=loss_function,optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm = 1.0), metrics=[metric_function])#,sample_weight_mode='temporal')
    # exit()
    return data_enc_model, enc_model, dec_model, model



def tag_tag_report_data(file_path, data_folder, image_folder):
    with open(file_path, 'r') as f:
        raw_data = f.read()
    lines = raw_data.split('\n')
    files=[]
    tags = []
    for line in lines[:-1]:
        filename = line.split(' ')[0]
        current_tags = []
        count = 1
        # print(line)
        # exit()
        # print(temp_tag.shape)
        # exit()
        for index in line.split(' ')[1:]:
            if index == '1':
                current_tags.append(count)
            count+=1
        # while len(current_tags) !=50:
        #     current_tags.append(0)
        files.append(filename)
        tags.append(current_tags)
    files, tags = (list(t) for t in zip(*sorted(zip(files, tags))))
    
    
    report_data = []
    report_labels = []
    i = 0
    remove_indexes = []
    image_data = []
    for file in files:
        current_image_file = os.path.join(image_folder, file+'.npy')
        current_file = os.path.join(data_folder, file+'.npy')
        if os.path.exists(current_file):
            current_data = np.load(current_file)
            current_data = np.expand_dims(current_data,1)
            report_data.append(current_data[:-1,:])
            report_labels.append(current_data[1:,:])
            current_feature = np.load(current_image_file)
            image_data.append(current_feature)
        else:
            remove_indexes.append(i)
        i+=1
    # exit()
    normal_vs_abnormal = []
    for file in files:
        current_data = np.load('/mnt/Data1/Preethi/XRay_Report_Generation/Data/tag_data_automatic/'+file+'.npy')
        if current_data[-1] == 0.0:
            normal_vs_abnormal.append(0)
        else:
            normal_vs_abnormal.append(1)
    normal_vs_abnormal = [i for j, i in enumerate(normal_vs_abnormal) if j not in remove_indexes]
    normal_vs_abnormal = np.asarray(normal_vs_abnormal, dtype = np.float32)
    tags = [i for j, i in enumerate(tags) if j not in remove_indexes]
    tags = np.expand_dims(np.asarray(tags, dtype = np.float32),2)
    report_data = np.asarray(report_data, dtype = np.float32)
    report_labels = np.asarray(report_labels, dtype = np.float32)
    image_data = np.asarray(image_data, dtype = np.float32)

    return tags, image_data, report_data, report_labels, normal_vs_abnormal


def load_data():
    inp_to_enc_model_data, inp_to_image_enc_model_data, inp_to_dec_model_data, inp_to_dec_model_labels, normal_vs_abnormal = tag_tag_report_data(tags_file_path, IU_7430_data_details_path, image_data_path)
    inp_to_dec_model_data, inp_to_dec_model_test_data, inp_to_dec_model_labels, inp_to_dec_model_test_labels, inp_to_enc_model_data, inp_to_enc_model_test_data, inp_to_image_enc_model_data, inp_to_image_enc_model_test_data, normal_vs_abnormal, normal_vs_abnormal_test = train_test_split(inp_to_dec_model_data, inp_to_dec_model_labels, inp_to_enc_model_data, inp_to_image_enc_model_data, normal_vs_abnormal, test_size = 500, random_state = 666)
    inp_to_dec_model_train_data, inp_to_dec_model_val_data, inp_to_dec_model_train_labels, inp_to_dec_model_val_labels, inp_to_enc_model_train_data, inp_to_enc_model_val_data, inp_to_image_enc_model_train_data, inp_to_image_enc_model_val_data, normal_vs_abnormal_train, normal_vs_abnormal_val = train_test_split(inp_to_dec_model_data, inp_to_dec_model_labels, inp_to_enc_model_data, inp_to_image_enc_model_data, normal_vs_abnormal, test_size = 500, random_state = 666)

    new_data_1 = []
    new_data_2 = []
    new_data_3 = []
    new_data_4 = []
    for i in range(normal_vs_abnormal_train.shape[0]):
        if normal_vs_abnormal_train[i] == 0:
            new_data_1.append(inp_to_dec_model_train_data[i])
            new_data_2.append(inp_to_dec_model_train_labels[i])
            new_data_3.append(inp_to_enc_model_train_data[i])
            new_data_4.append(inp_to_image_enc_model_train_data[i])
    inp_to_dec_model_train_data = np.asarray(new_data_1, dtype = np.float32)
    inp_to_dec_model_train_labels = np.asarray(new_data_2, dtype = np.float32)
    inp_to_enc_model_train_data = np.asarray(new_data_3, dtype = np.float32)
    inp_to_image_enc_model_train_data = np.asarray(new_data_4, dtype = np.float32)

    new_data_1 = []
    new_data_2 = []
    new_data_3 = []
    new_data_4 = []
    for i in range(normal_vs_abnormal_val.shape[0]):
        if normal_vs_abnormal_val[i] == 0:
            new_data_1.append(inp_to_dec_model_val_data[i])
            new_data_2.append(inp_to_dec_model_val_labels[i])
            new_data_3.append(inp_to_enc_model_val_data[i])
            new_data_4.append(inp_to_image_enc_model_val_data[i])
    inp_to_dec_model_val_data = np.asarray(new_data_1, dtype = np.float32)
    inp_to_dec_model_val_labels = np.asarray(new_data_2, dtype = np.float32)
    inp_to_enc_model_val_data = np.asarray(new_data_3, dtype = np.float32)
    inp_to_image_enc_model_val_data = np.asarray(new_data_4, dtype = np.float32)

    class_count = np.zeros((1000,))
    total_count = 0
    for i in range(inp_to_dec_model_train_labels.shape[0]):
        for j in range(259):
            if inp_to_dec_model_train_labels[i,j,0] !=0.0:
                class_count[int(inp_to_dec_model_train_labels[i,j,0])] += 1
                total_count += 1
    class_weights = np.zeros((1000,))
    for i in range(1000):
        if class_count[i] != 0:
            class_weights[i] = total_count / (1000*class_count[i])


    return inp_to_enc_model_train_data, inp_to_dec_model_train_data, inp_to_image_enc_model_train_data, inp_to_dec_model_train_labels, inp_to_enc_model_val_data, inp_to_dec_model_val_data, inp_to_image_enc_model_val_data, inp_to_dec_model_val_labels, inp_to_enc_model_test_data, inp_to_dec_model_test_data, inp_to_image_enc_model_test_data, inp_to_dec_model_test_labels, class_weights, normal_vs_abnormal_test


def open_metric_files():
    per_batch_train_metrics_file = open(result_files_path + '/training_per_batch_metrics' + '.txt', 'a')
    per_epoch_train_metrics_file = open(result_files_path + '/training_per_epoch_metrics' + '.txt', 'a')
    per_epoch_val_metrics_file = open(result_files_path + '/test_metrics' + '.txt', 'a')

    return per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file

def train():
    data_enc_model, enc_model, dec_model, final_model = make_model()
    final_model.summary()
    exit()
    inp_to_enc_model_train_data, inp_to_dec_model_train_data, inp_to_image_enc_model_train_data, inp_to_dec_model_train_labels, inp_to_enc_model_val_data, inp_to_dec_model_val_data, inp_to_image_enc_model_val_data, inp_to_dec_model_val_labels, _, _, _, _, class_weights,_ = load_data()
    # class_weights = dict(enumerate(class_weights))
    # print('yaaaay')
    # exit()
    inp_to_dec_model_train_labels = tf.keras.utils.to_categorical(inp_to_dec_model_train_labels, num_classes=target_vocab_size)
    inp_to_dec_model_val_labels = tf.keras.utils.to_categorical(inp_to_dec_model_val_labels, num_classes=target_vocab_size)
    save_model_list=[]
    test_e = 0
    min_loss = 100.0
    num_of_batches = int(math.ceil(inp_to_dec_model_train_data.shape[0]/batch_size))
    per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file = open_metric_files()
    for e in range(num_epochs):
        per_batch_train_metrics_file, per_epoch_train_metrics_file, per_epoch_val_metrics_file = open_metric_files()
        inp_to_image_enc_model_train_data, inp_to_enc_model_train_data, inp_to_dec_model_train_data, inp_to_dec_model_train_labels = shuffle(inp_to_image_enc_model_train_data, inp_to_enc_model_train_data, inp_to_dec_model_train_data, inp_to_dec_model_train_labels, random_state = 2)
        per_epoch_train_loss = 0.0
        per_epoch_train_acc = 0.0
        for batch_num in range(num_of_batches):
        # for batch_num in range(2):
            batch_X_train_inp_to_enc_model = np.zeros((batch_size, 16))
            batch_X_train_inp_to_image_enc_model = np.zeros((batch_size, 10, 10, 512))
            batch_X_train_inp_to_dec_model = np.zeros((batch_size, 259))
            batch_y_train_inp_to_dec_model = np.zeros((batch_size, 259, target_vocab_size))
            batch_sample_Weights = np.zeros((batch_size,259))
            b = 0
            for j in range(batch_num*batch_size, min((batch_num+1)*batch_size, inp_to_dec_model_train_data.shape[0])):
                batch_X_train_inp_to_enc_model[b,:] = inp_to_enc_model_train_data[j,:,0]
                batch_X_train_inp_to_image_enc_model[b,:,:,:] = inp_to_image_enc_model_train_data[j,:,:,:]
                batch_X_train_inp_to_dec_model[b,:] = inp_to_dec_model_train_data[j,:,0]
                # batch_X_train_inp_to_dec_model[b,:] = 2.0       
                batch_y_train_inp_to_dec_model[b, :, :] = inp_to_dec_model_train_labels[j,:,:]
                b += 1
            sample_classes = np.argmax(batch_y_train_inp_to_dec_model,axis = 2)
            # print(batch_y_train_inp_to_dec_model.shape)
            # print(sample_classes.shape)
            for bi in range(batch_size):
                for bj in range(259):
                    batch_sample_Weights[bi,bj] = class_weights[int(sample_classes[bi,bj])]
            # exit()
            per_batch_train_loss, per_batch_train_acc = final_model.train_on_batch([batch_X_train_inp_to_enc_model, batch_X_train_inp_to_image_enc_model, batch_X_train_inp_to_dec_model], batch_y_train_inp_to_dec_model)#, sample_weight = batch_sample_Weights)
            
            print('epoch_num: %d, batch_num: %d, loss: %f, class_wise_accuracy: %s\n' % (e, batch_num, per_batch_train_loss, per_batch_train_acc))
            # exit()
            write_metric_files([per_batch_train_metrics_file], [[e, batch_num, per_batch_train_loss, per_batch_train_acc]])

            per_epoch_train_loss += per_batch_train_loss
            per_epoch_train_acc += per_batch_train_acc

        per_epoch_val_loss, per_epoch_val_acc = 0.0, 0.0
        for i in range(inp_to_dec_model_val_data.shape[0]):
            # exit()
            # temp_ip = np.zeros((1,259))
            # temp_ip[0,:] = 2.0
            test_sample_weight = np.zeros((1,259))
            test_current_classes = np.argmax(inp_to_dec_model_val_labels[i:i+1,:,:], axis=2)
            for bi in range(259):
                test_sample_weight[0,bi] = class_weights[int(test_current_classes[0,b])]
            curr_val_loss, curr_val_acc = final_model.test_on_batch([inp_to_enc_model_val_data[i:i+1,:,0], inp_to_image_enc_model_val_data[i:i+1,:,:,:],inp_to_dec_model_val_data[i:i+1,:,0]], inp_to_dec_model_val_labels[i:i+1,:,:])#, sample_weight = test_sample_weight)
            # curr_val_loss, curr_val_acc = final_model.test_on_batch([inp_to_enc_model_val_data[i:i+1,:,0], temp_ip], inp_to_dec_model_val_labels[i:i+1,:,:])
            per_epoch_val_loss += curr_val_loss
            per_epoch_val_acc += curr_val_acc
        
        per_epoch_val_loss /= inp_to_dec_model_val_data.shape[0]
        per_epoch_val_acc /= inp_to_dec_model_val_data.shape[0]
        test_e+=1
        print('---------------------------------------------------------------------\n')
        print('test_num: %d, loss: %f, \nclass_wise_accuracy: %s\n' % (test_e, per_epoch_val_loss, per_epoch_val_acc))

        if per_epoch_val_loss < min_loss:
            save_model_list.append(test_e)
            final_model.save_weights(model_weights_path + '/final_model_' + str(test_e)+'.h5')
            data_enc_model.save_weights(model_weights_path + '/data_enc_model_' + str(test_e)+'.h5')
            enc_model.save_weights(model_weights_path + '/enc_model_' + str(test_e)+'.h5')
            dec_model.save_weights(model_weights_path + '/dec_model_' + str(test_e)+'.h5')
            if len(save_model_list)>10:
                del_model_count = save_model_list.pop(0)
                os.remove(model_weights_path + '/final_model_' + str(del_model_count)+'.h5')
                os.remove(model_weights_path + '/data_enc_model_' + str(del_model_count)+'.h5')
                os.remove(model_weights_path + '/enc_model_' + str(del_model_count)+'.h5')
                os.remove(model_weights_path + '/dec_model_' + str(del_model_count)+'.h5')
            # prev_epoch = 0
            print('Model loss improved from %f to %f. Saving Weights\n'%(min_loss, per_epoch_val_loss))
            min_loss = per_epoch_val_loss
        else:
            print('Best loss: %f\n'%(min_loss))
        print('----------------------------------------------------------------------\n')
        write_metric_files([per_epoch_val_metrics_file], [[min_loss, test_e, per_epoch_val_loss, per_epoch_val_acc]])


        per_epoch_train_loss = per_epoch_train_loss / num_of_batches
        per_epoch_train_acc = per_epoch_train_acc / num_of_batches
        write_metric_files([per_epoch_train_metrics_file], [[e, per_epoch_train_loss, per_epoch_train_acc]])

def test():
    _,enc_model, dec_model, final_model = make_model()
    final_model.load_weights(model_weights_path+'/final_model_51.h5')
    _, _, _, _, _, _, _, _, inp_to_enc_model_test_data, inp_to_dec_model_test_data, inp_to_image_enc_model_test_data, inp_to_dec_model_test_label,_, normal_vs_abnormal_test = load_data()
    test_acc1 = 0.0
    test_acc2 = 0.0
    test_acc3 = 0.0
    test_acc4 = 0.0
    test_acc5 = 0.0
    count=0
    with open('test.txt','w') as f:
        do_nothing=1
    for i in range(inp_to_dec_model_test_data.shape[0]):
        print('Testing '+str(i+1)+' sample out of '+str(inp_to_dec_model_test_data.shape[0]))
        ref2=[]
        # print(inp_to_dec_model_test_label.shape)
        for j in range(259):
            #to remove '.', 'end character' and 'pad'
            if (inp_to_dec_model_test_label[i,j,0] != 2) and (inp_to_dec_model_test_label[i,j,0] != 1) and (inp_to_dec_model_test_label[i,j,0] != 0):
                ref2.append(str(int(inp_to_dec_model_test_label[i,j,0])))
        # print(ref2)
        if len(ref2)>=5 and normal_vs_abnormal_test[i] == 0:
            index = 2
            text_input_list = []
            while (index != 1) and (len(text_input_list)<260):
                text_input_list.append(index)
                inp_text = np.asarray(text_input_list)
                inp_text = np.expand_dims(inp_text,0)
                out = final_model.predict([inp_to_enc_model_test_data[i:i+1,:,0], inp_to_image_enc_model_test_data[i:i+1,:,:,:], inp_text])
                out = np.squeeze(out,0)
                out = np.argmax(out,1)
                index = out[-1]
            # print(text_input_list)
            ref=[]
            for j in range(len(text_input_list)):
                if (text_input_list[j]!=2):
                    ref.append(str(text_input_list[j]))
            # print(ref)
            with open('test.txt','a') as f:
                for abc in range(len(ref)):
                    f.write(str(ref[abc]))
                    f.write(' ')
                f.write('\n')
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
    print('Bleu 1: '+str(test_acc1)+'\nBleu 2: '+str(test_acc2)+'\nBleu 3: '+str(test_acc3)+'\nBleu 4: '+str(test_acc4)+'\nBleu 5: '+str(test_acc5))

    

#################################### Global Arguments #######################################
# loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, reduction='none')
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction='none')
#############################################################
batch_size = 128
num_epochs = 10000
target_vocab_size=1000
number_of_tags = 573
enc_num_layers = 1
enc_d_model = 256
enc_num_heads = 8
enc_dff = enc_d_model
enc_rate = 0.1
dec_num_layers = 1
dec_d_model = 256
dec_num_heads = 8
dec_dff = dec_d_model
maximum_position_encoding = 300
dec_rate = 0.1
data_enc_num_layers = 1
data_enc_d_model = 256
data_enc_num_heads = 1
data_enc_dff = data_enc_d_model
data_enc_rate = 0.1
#############################################################
server_path = "/mnt/Data1/Preethi/XRay_Report_Generation"
# server_path = "/home/ada/Preethi/XRay_Report_Generation"
#############################################################
IU_7430_data_details_path = os.path.join(server_path, 'Data', 'text_data_img_caption')
tags_file_path = os.path.join(server_path, 'Data', 'tags_automatic_predicted_trim.txt')
result_files_path = os.path.join(server_path, 'Code', '6_report_generation', 'Results_trans_ranking_image')
model_weights_path = os.path.join(server_path, 'Code', '6_report_generation', 'Model_Weights_ranking_image')
image_data_path = os.path.join(server_path, 'Data', 'IU_features_512')
actual_tags_path = os.path.join(server_path, 'Data', 'tag_data_automatic_trim')
#################################### Global Arguments #######################################
if not os.path.exists(model_weights_path):
    os.makedirs(model_weights_path)
if not os.path.exists(result_files_path):
    os.makedirs(result_files_path)

train()
# test()