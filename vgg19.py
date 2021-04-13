# https://tensorlayer.readthedocs.io/en/1.10.0/_modules/tensorlayer/models/vgg19.html
# https://github.com/udacity/deep-learning/blob/master/transfer-learning/tensorflow_vgg/vgg19.py
import os
import tensorflow as tf

import numpy as np

# CUSTOMISED VGG MODEL 
class vgg19():

		# retrieves filter from dataset, applies bias 
	def conv_layer(self, input, layer_name):
			# (3, 3, 64, 64) filter
			fil = tf.constant(self.data_dict[layer_name][0])
			# input is previous tensor 
			layer = tf.nn.conv2d(input, fil, strides=[1, 1, 1, 1], padding='SAME')
			# (64,) bias 
			layer = tf.nn.bias_add(layer, tf.constant(self.data_dict[layer_name][1]))
			# apply relu to standardise 
			return tf.nn.relu(layer)

	# model weights: path to load npy 
	def __init__(self, input_img):

		# latin needed to import weights!
		# else load of rubbish 
		self.data_dict = np.load('./vgg19.npy', encoding='latin1', allow_pickle=True).item()

		VGG_MEAN = [103.939, 116.779, 123.68]

		# splits (1, 224, 224, 3) into its 3 channels each (1, 224, 224, 1)
		r_channel, g_channel, b_channel = tf.split(input_img, 3, 3)
		
		# Convert RGB to BGR, subtract training dataset mean values 
		# VGG19 expects BGR
		bgr = tf.concat([b_channel - VGG_MEAN[0], g_channel - VGG_MEAN[1],
										r_channel - VGG_MEAN[2],], axis=3)

		# no FC layers
		self.conv1_1 = self.conv_layer(input_img, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
		# tf.nn.avg_pool or tf.nn.max_pool
		self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

		self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
		self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
		self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

		self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
		self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
		self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
		self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
		self.pool3 = tf.nn.max_pool(self.conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

		self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
		self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
		self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
		self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
		self.pool4 = tf.nn.max_pool(self.conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

		self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
		self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
		self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
		self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
		self.pool5 = tf.nn.max_pool(self.conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')