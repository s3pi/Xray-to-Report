import tensorflow as tf

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



class Narrow_MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(Narrow_MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    # assert d_model % self.num_heads == 0
    
    # self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model*num_heads, kernel_initializer = tf.keras.initializers.lecun_normal())
    self.wk = tf.keras.layers.Dense(d_model*num_heads, kernel_initializer = tf.keras.initializers.lecun_normal())
    self.wv = tf.keras.layers.Dense(d_model*num_heads, kernel_initializer = tf.keras.initializers.lecun_normal())
    
    self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.lecun_normal())
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, d_model).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, d_model)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v_ip, k_ip, q_ip, mask=None):
    batch_size = tf.shape(q_ip)[0]
    
    q = self.wq(q_ip)  # (batch_size, seq_len, d_model * num_heads)
    k = self.wk(k_ip)  # (batch_size, seq_len, d_model * num_heads)
    v = self.wv(v_ip)  # (batch_size, seq_len, d_model * num_heads)
    # print(q.shape)
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, d_model)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, d_model)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, d_model)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, d_model)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, d_model)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model*self.num_heads))  # (batch_size, seq_len_q, d_model * num_head)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', kernel_initializer = tf.keras.initializers.lecun_normal()),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.lecun_normal())])  # (batch_size, seq_len, d_model)
