# IMAGE TRANSFORMATION NETWORK
# Sturcture based on https://arxiv.org/pdf/1709.04111.pdf and https://arxiv.org/pdf/1603.08155.pdf

import tensorflow as tf, pdb

def conv(self_v, net, num_filters, filter_size, num_str, relu=True, batchnorm=True):
    net = tf.nn.conv2d(input=net, filters=self_v, strides=[1, num_str, num_str, 1], padding='SAME')
    if relu:
        net = tf.nn.relu(net)
    if batchnorm:
      net = batch_norm(net)
    return net

def conv_tranpose(self_v, net, num_filters, filter_size, num_str, relu=True, batchnorm=True):    
    batch, rows, cols, ch = [i for i in net.get_shape()]

    new_rows, new_cols = rows * num_str, cols * num_str
    
    new_shape = [batch, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    net = tf.nn.conv2d_transpose(net, self_v, tf_shape, [1,num_str,num_str,1], padding='SAME')
    if relu:
      net = tf.nn.relu(net)
    if batchnorm:
      net = batch_norm(net)
    return net

def batch_norm(x):
    mean, var = tf.nn.moments(x, axes=[1, 2, 3])

    mean = tf.math.reduce_mean(mean)
    var = tf.math.reduce_std(var)**2

    return tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-5)

def residual_b(self_v, net, filter_size=3):
    h = conv(self_v, net, 128, filter_size, 1)
    # self + conv
    return net + conv(self_v, h, 128, filter_size, 1, relu=False)

def init_conv(net, out_ch, filter_size, name, transpose=False):

  batch, rows, cols, ch = [i for i in net.get_shape()]

  if not transpose:
      transp_shape = [filter_size, filter_size, ch, out_ch]
  else:
      transp_shape = [filter_size, filter_size, out_ch, cols]

  # random values with 0.001 standard dev 
  init_val = tf.Variable(tf.random.truncated_normal(transp_shape, 0.001), dtype=tf.float32)
  return tf.Variable(init_val, name=name)

def init_vars(net, out_ch, filter_size, name, transpose=False):

  #print("shape:")
  #print(net.get_shape())

  batch, rows, cols, ch = [i for i in net.get_shape()]

  if not transpose:
      init_shape = [filter_size, filter_size, ch, out_ch]
  else:
      init_shape = [filter_size, filter_size, out_ch, cols]

  init_weight = tf.Variable(tf.random.truncated_normal(init_shape, 0.001), dtype=tf.float32)
  return tf.Variable(init_weight, name=name)

class transformer():
  # Need to initialise variables to save the graph 
  def __init__(self, image):
    self.conv1 = init_vars(image, 32, 9, "t_conv1_w")
    self.conv2 = init_vars(self.conv1, 64, 4, "t_conv2_w")
    self.conv3 = init_vars(self.conv2, 128, 4, "t_conv3_w")

    self.resid1 = init_vars(self.conv3, 128, 3, "R_conv1_w")
    self.resid2 = init_vars(self.resid1, 128, 3, "R_conv2_w")
    self.resid3 = init_vars(self.resid2, 128, 3, "R_conv3_w")
    self.resid4 = init_vars(self.resid3, 128, 3, "R_conv4_w")
    self.resid5 = init_vars(self.resid4, 128, 3, "R_conv5_w")

    self.conv_t1 = init_vars(self.resid5, 64, 4, "t_dconv1_w", transpose=True)
    self.conv_t2 = init_vars(self.conv_t1, 32, 4, "t_dconv2_w", transpose=True)
    self.conv_t3 = init_vars(self.conv_t2, 3, 9, "t_dconv3_w", transpose=True)

  def __call__(self, image):
    # convolution layers 
    self.conv1 = conv(self.conv1, image, 32, 9, 1)
    self.conv2 = conv(self.conv2, self.conv1, 64, 3, 2)
    self.conv3 = conv(self.conv3, self.conv2, 128, 3, 2)
    # residual layers 
    self.resid1 = residual_b(self.resid1, self.conv3, 3)
    self.resid2 = residual_b(self.resid2, self.resid1, 3)
    self.resid3 = residual_b(self.resid3, self.resid2, 3)
    self.resid4 = residual_b(self.resid4, self.resid3, 3)
    self.resid5 = residual_b(self.resid5, self.resid4, 3)
    # convolutional transpose layers 
    self.conv_t1 = conv_tranpose(self.conv_t1, self.resid5, 64, 3, 2)
    self.conv_t2 = conv_tranpose(self.conv_t2, self.conv_t1, 32, 3, 2)
    self.conv_t3 = conv_tranpose(self.conv_t3, self.conv_t2, 3, 9, 1)
    # output node 
    output = tf.multiply((tf.tanh(self.conv_t3) + 1), tf.constant(127.5, tf.float32, shape=self.conv_t3.get_shape()), name='output')
    return output
