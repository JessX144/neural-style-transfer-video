# TRAINS TRANSFORMER NET 
import tensorflow as tf
import os
from transformer import transformer
from vgg19 import vgg19
import cv2
import numpy as np

style_layers = ['conv1_1',
                'conv2_1',
                'conv3_1', 
                'conv4_1', 
                'conv5_1']

style_weights = [1.0,
                0.8,
                0.5, 
                0.3, 
                0.1]

content_layers = ['conv4_2']
content_weights = [0.5]

learn_rate = 1e-3
var_weight = 10e-4
epoch = 2

list = os.listdir('./training_dataset2') # dir is your directory path
num_data = len(list)

#iter = int(num_data / b_size)
# frames in one vid 
b_size = 5

VGG_MEAN = [103.939, 116.779, 123.68]

def preprocess_img(img):

  imgpre = img.copy()
  imgpre.resize(224, 224, 3)
  imgpre = np.asarray(imgpre, dtype=np.float32)
  return imgpre

# gets all content images of certain name 
def get_content_imgs(name):
  
  imgs = np.zeros((b_size, 224, 224, 3), dtype=np.float32)

  for filename in os.listdir('./input_images/'):
    if (name in filename.split(".")[0]):
      img = cv2.imread(os.path.join('./input_images/',filename))
      img = preprocess_img(img)
      imgs[0] = img
  return imgs

def get_train_imgs(name):
  
  imgs = np.zeros((num_data, 224, 224, 3), dtype=np.float32)

  for filename in os.listdir('./training_dataset2/'):
    ind = 0;
    if (name in filename.split(".")[0]):
      img = cv2.imread(os.path.join('./training_dataset2/',filename))
      img = preprocess_img(img)
      imgs[ind] = img
      ind += 1
  return imgs

def get_style_img(img):
  img = cv2.imread('./style_images/' + str(img) + '.jpg', cv2.IMREAD_COLOR)
  img = preprocess_img(img)
  return img

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
  input = tf.placeholder(tf.float32, shape=[b_size, 224, 224, 3], name='input')
  #print('img2 shape: {}'.format(input.shape))
  style_img = tf.placeholder(tf.float32, shape=[b_size, 224, 224, 3], name='style_img')

  # initialise net 
  trans_net = transformer(input)

  output = trans_net(input)
  vgg_outp = vgg19(output)

  vgg_style = vgg19(style_img)
  vgg_content = vgg19(input)
  
  #for style_output in style_layers:
  style_outputs = [gram_matrix(vgg_style.__dict__[style_output]) for style_output in style_layers]
  style_targets = [gram_matrix(vgg_outp.__dict__[style_output]) for style_output in style_layers]

  content_outputs = [(vgg_content.__dict__[content_output]) for content_output in content_layers]
  content_targets = [(vgg_outp.__dict__[content_output]) for content_output in content_layers]

  style_loss = tf.zeros(b_size, tf.float32)
  for i in range(len(style_outputs)):
      style_loss += style_weights[i] * tf.reduce_mean(tf.subtract(style_targets[i], style_outputs[i]) ** 2, [1, 2])
  
  content_loss = tf.zeros(b_size, tf.float32)
  for i in range(len(content_layers)):
    content_loss += content_weights[i] * tf.reduce_mean(tf.subtract(content_targets[i], content_outputs[i]) ** 2, [1, 2, 3])

  var_loss = var_weight * total_variation_loss(output)
  print('var loss: {}'.format(var_loss))
  print('style loss: {}'.format(style_loss))
  print('content loss: {}'.format(content_loss))
  total_loss = style_loss + content_loss + var_loss

  train = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

  tf.global_variables_initializer().run()

  style = get_style_img('lions')
  # dict must be array elements 
  style_np = [style for x in range(b_size)]
  print(style_np)
  print('len: {}'.format(len(style_np)))

  # gets all content images you need - video frames
  imgs = get_train_imgs('vid_frames')
  print('img length: {}'.format(len(imgs)))

  for e in range(0, epoch):
    inp_imgs = np.zeros((b_size, 224, 224, 3), dtype=np.float32)

    iter = int(num_data / b_size)

    for i in range(iter):
      for j in range(b_size):
        #print('j: {}'.format(j))
        #print('i: {}'.format(i))
        inp_imgs[j] = imgs[i * b_size + j]
      dict = {input: inp_imgs, style_img: style_np}
      loss, _ = sess.run([total_loss, train], feed_dict=dict)
      print('[iter {}/{}] loss: {}'.format(i + 1, iter, loss[0]))