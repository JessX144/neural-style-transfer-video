# IMAGE TRANSFORMATION NETWORK
# Sturcture based on https://arxiv.org/pdf/1709.04111.pdf and https://arxiv.org/pdf/1603.08155.pdf

import tensorflow as tf, pdb

norm = "i"

def batch_norm(x):
		mean, var = tf.nn.moments(x, axes=[1, 2, 3])

		b_var = tf.reduce_mean(var)
		b_mean = tf.reduce_mean(mean)

		return tf.nn.batch_normalization(x, b_mean, b_var, 0, 1, 1e-5)

def inst_norm(x):
	mean, var = tf.nn.moments(x, axes=[1, 2])

	i_var = tf.reduce_mean(var)
	i_mean = tf.reduce_mean(mean)

	return tf.nn.batch_normalization(x, i_mean, i_var, 0, 1, 1e-10)

def conv(self_v, net, num_filters, filter_size, num_str, relu=True):
		f_size = int(filter_size/2)
		# should help with border effect 
		net = tf.pad(net, [[0,0], [f_size, f_size], [f_size, f_size], [0,0]], mode='REFLECT')
		net = tf.nn.conv2d(net, self_v, strides=[1, num_str, num_str, 1], padding='VALID')

		if (relu):
			net = tf.nn.relu(net)

		if (norm == "b"):
			net = batch_norm(net)
		elif (norm == "i"):
			net = inst_norm(net)
		return net

def conv_tranpose(self_v, net, num_filters, num_str, relu=True, normalise=True):		
		batch, rows, cols, ch = [i for i in net.get_shape()]
		new_rows = rows * num_str
		new_cols = cols * num_str
		new_shape = [batch, new_rows, new_cols, num_filters]

		net = tf.nn.conv2d_transpose(net, self_v, new_shape, [1,num_str,num_str,1], padding='SAME')
		
		if (relu):
			net = tf.nn.relu(net)

		if (normalise):
			if (norm == "b"):
				net = batch_norm(net)
			elif (norm == "i"):
				net = inst_norm(net)
		return net

class res():
	def __init__(self, x, i, j, n):
		self.w1 = init_vars(x, i, j, n)
		self.w2 = init_vars(x, i, j, n)
	def __call__(self_v, im):
		r = tf.nn.relu(batch_norm(tf.nn.conv2d(im, self_v.w1, [1, 1, 1, 1], 'SAME')))
		r = batch_norm(tf.nn.conv2d(r, self_v.w2, [1, 1, 1, 1], 'SAME'))
		return im + r

def init_vars(net, out_ch, filter_size, name, transpose=False):
	batch, rows, cols, ch = [i for i in net.get_shape()]
	if transpose:
			init_shape = [filter_size, filter_size, out_ch, cols]
	else:
		init_shape = [filter_size, filter_size, ch, out_ch]
	return tf.Variable(tf.random.truncated_normal(init_shape, stddev=0.001), name=name)

class transformer():
	# Need to initialise variables to save the graph 
	def __init__(self, image):

		self.conv1 = init_vars(image, 32, 9, "t_conv1_w")
		self.conv2 = init_vars(self.conv1, 64, 4, "t_conv2_w")
		self.conv3 = init_vars(self.conv2, 128, 4, "t_conv3_w")

		self.resid1 = res(self.conv3, 128, 3, "R1_conv1_w")
		self.resid2 = res(self.conv3, 128, 3, "R2_conv1_w")
		self.resid3 = res(self.conv3, 128, 3, "R3_conv1_w")
		self.resid4 = res(self.conv3, 128, 3, "R4_conv1_w")
		self.resid5 = res(self.conv3, 128, 3, "R5_conv1_w")

		self.conv_t1 = init_vars(self.resid5.w1, 64, 4, "t_dconv1_w", transpose=True)
		self.conv_t2 = init_vars(self.conv_t1, 32, 4, "t_dconv2_w", transpose=True)
		self.conv_t3 = init_vars(self.conv_t2, 3, 9, "t_dconv3_w", transpose=True)

	def __call__(self, image):

		# removes border effect
		image = tf.pad(image, [[0,0], [10,10], [10,10],[0,0]], mode='REFLECT')

		# convolution layers 
		image = conv(self.conv1, image, 32, 9, 1)
		image = conv(self.conv2, image, 64, 3, 2)
		image = conv(self.conv3, image, 128, 3, 2)

		# residual layers 
		image = self.resid1(image)
		image = self.resid2(image)
		image = self.resid3(image)
		image = self.resid4(image)
		image = self.resid5(image)

		# convolutional transpose layers 
		image = conv_tranpose(self.conv_t1, image, 64, 2)
		image = conv_tranpose(self.conv_t2, image, 32, 2)
		image = conv_tranpose(self.conv_t3, image, 3, 1, relu=False, normalise=False)

		# ensures output is in range of 0-255
		# different methods of doing so 
		output = tf.multiply((tf.tanh(image) + 1), tf.constant(127.5, tf.float32, shape=image.get_shape()), name='output') 
		# output = tf.Variable(tf.nn.tanh(image) * 150 + 255./2, name='output')
		
		# remove padding 
		height = tf.shape(output)[1]
		width = tf.shape(output)[2]
		output = tf.slice(output, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

		# (1, 224, 224, 3)
		return output
