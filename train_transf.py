# TRAINS TRANSFORMER NET 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformer import transformer
from vgg19 import vgg19
import cv2
import numpy as np
from PIL import Image

from variables import b_size, epoch

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

learn_rate = 1e-2
var_weight = 10e-4

list = os.listdir('./training_dataset2') # dir is your directory path
num_data = len(list)

VGG_MEAN = [103.939, 116.779, 123.68]

style_name = 'lions'

def preprocess_img(img):

	imgpre = img.copy()
	imgpre = imgpre.resize((224, 224))
	imgpre = np.asarray(imgpre, dtype=np.float32)
	
	return imgpre

def get_train_imgs(name):
  
	imgs = np.zeros((num_data, 224, 224, 3), dtype=np.float32)
	ind = 0;
	for filename in os.listdir('./training_dataset2/'):
		if (name in filename.split(".")[0]):
			img = Image.open(os.path.join('./training_dataset2/',filename)).convert('RGB')
			print(filename)
			img = preprocess_img(img)
			imgs[ind] = img
			ind += 1
	return imgs

def get_style_img(img):
	img = Image.open('./style_images/' + str(img) + '.jpg').convert('RGB')
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

# Saver can't operate on GPU
input = tf.placeholder(tf.float32, shape=[b_size, 224, 224, 3], name='input')
tf.identity(input, name="input")
trans_net = transformer(input)
saver = tf.train.Saver()
saver = tf.train.Saver(restore_sequentially=True)

with tf.device('/gpu:0'):
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
  #print('var loss: {}'.format(var_loss))
  #print('style loss: {}'.format(style_loss))
  #print('content loss: {}'.format(content_loss))
  total_loss = style_loss + content_loss + var_loss

  train = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

	ckpt_directory = './checkpts/{}/'.format(style_name)
	if not os.path.exists(ckpt_directory):
		os.makedirs(ckpt_directory)

	tf.global_variables_initializer().run()	

	style = get_style_img(style_name)
  # dict must be array elements 
	style_np = [style for x in range(b_size)]
	#print(style_np)
	print('len: {}'.format(len(style_np)))

  # gets all content images you need - video frames
	imgs = get_train_imgs('frame')
	print('img length: {}'.format(len(imgs)))

	for e in range(epoch):
		inp_imgs = np.zeros((b_size, 224, 224, 3), dtype=np.float32)

		iter = int(num_data / b_size)

		for i in range(iter):
			for j in range(b_size):
				inp_imgs[j] = imgs[i * b_size + j]
			dict = {input: inp_imgs, style_img: style_np}
			loss, _ = sess.run([total_loss, train], feed_dict=dict)
			print('[iter {}/{}] loss: {}'.format(i + 1, iter, loss[0]))
	saver.save(sess, ckpt_directory + style_name, global_step=e)

		#for i in range(4):
		#	cv2.imwrite('./output_images/{}_output.jpg'.format(i), inp_imgs[i])