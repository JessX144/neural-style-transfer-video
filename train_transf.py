# TRAINS TRANSFORMER NET 
import tensorflow as tf

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

style_weights = {'block1_conv1': 1.,
                 'block2_conv1': 0.8,
                 'block3_conv1': 0.5,
                 'block4_conv1': 0.3,
                 'block5_conv1': 0.1}

content_ouputs = ['block4_conv2']
content_weights = {'block4_conv2': 0.5}

learn_rate = 1e-3

var_weight = 10e-4

def gram_matrix(input_tensor):
  # output[b,c,d] = sum_w input_tensor[b,i,j,c] * input_tensor[b,i,j,c]
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

def total_variation_loss(image):
  x_deltas = image[:,:,1:,:] - image[:,:,:-1,:]
  y_deltas = image[:,1:,:,:] - image[:,:-1,:,:]
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

with tf.device('/gpu:0'):

  input = tf.placeholder(tf.float32, shape=[batchsize, 224, 224, 3], name='input')
  style_img = tf.placeholder(tf.float32, shape=[batchsize, 224, 224, 3], name='target')

  output = transformer(input)
  vgg_outp = vgg19(output)

  vgg_style = vgg19(style_img)
  vgg_content = vgg19(input)
  
  style_outputs = [gram_matrix(vgg_style.style_output) for style_output in style_layers]
  style_targets = [gram_matrix(vgg_outp.style_output) for style_output in style_layers]

  content_outputs = [(vgg_content.content_output) for content_output in content_layers]
  content_targets = [(vgg_outp.content_output) for content_output in content_layers]
  
  style_loss = tf.zeros(batchsize, tf.float32)
  for i in style_outputs.len():
      style_loss += style_weights[i] * tf.reduce_mean(tf.subtract(style_targets[i], style_outputs[i]) ** 2, [1, 2])
  
  content_loss = tf.zeros(batchsize, tf.float32)
  for i in content_layers.len():
    content_loss += content_weights[i] * tf.reduce_mean(tf.subtract(content_targets[i], content_outputs[i]) ** 2, [1, 2])

  var_loss = var_weight * total_variation_loss(output)

  total_loss = style_loss + content_loss + var_loss

  train = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)

