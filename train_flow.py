# TRAINS TRANSFORMER NET - takes into account optical flow 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformer import transformer
from vgg19 import vgg19
import cv2
import numpy as np
from PIL import Image
import math
import time
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--style', '-s', type=str)
parser.add_argument('--norm', '-n', type=str, default="b")
args = parser.parse_args()


style_layers = ['conv1_1',
								'conv2_1',
								'conv3_1', 
								'conv4_1', 
								'conv5_1']
content_layers = ['conv4_2']

norm = args.norm 
epoch = 2
b_size = 1 # optical flow, do not take batches of images 
style_weight = 1e0
content_weight = 1e0
learn_rate = 1e-3
var_weight = 10e-4
temporal_weight = 1e-4 # values are quite large, ~1/5 of style loss  

tr = './input_images/bo/'
list = os.listdir('./input_images/bo') # dir is your directory path
num_data = len(list)

progress = './test_output/progress_flow/'

prev_im = np.zeros([b_size, 224, 224, 3], np.float32)
prev_im_stylised = np.zeros([b_size, 224, 224, 3], np.float32)
hsv = np.zeros((224,224,3))
hsv[...,1] = 255

def preprocess_img(img):

	imgpre = img.copy()
	imgpre = imgpre.resize((224, 224))
	imgpre = np.asarray(imgpre, dtype=np.float32)
	
	return imgpre


def unprocess_img(img, input_shape):
	img = img[0]
	# remove padding
	img = img[10:-10,10:-10,:]

	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	im = Image.fromarray(np.uint8(img))
	# resample NEAREST, BILINEAR, BICUBIC, ANTIALIAS 
	# filters for when resizing, change num pixels rather than resize 
	im = im.resize((input_shape[1], input_shape[0]), resample=Image.BILINEAR)
	im = np.array(im)
	return im

def get_train_imgs(name):
	imgs = []
	# cannot give all imgs, memory error, use list 
	for filename in list:
		if (name in os.path.splitext(filename)[0]):
			imgs.append(tr + filename)
	return imgs

def get_style_img(img):
	img = Image.open('./style_images/' + str(img) + '.jpg').convert('RGB')
	img = preprocess_img(img)
	return img

def gram_matrix(x):
		b, h, w, ch = x.get_shape().as_list()
		features = tf.reshape(x, [b, h*w, ch])
		gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32)
		return gram

# warps an image according to its flow, predicts its movement 
# from OpenCV
def warp_flow(img, flow):
		h, w = flow.shape[:2]
		flow = -flow
		flow[:,:,0] += np.arange(w)
		flow[:,:,1] += np.arange(h)[:,np.newaxis]
		res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
		return res

def get_flow_weights_bounds(flow, thresh): 

		x_dim = flow.shape[1]
		y_dim = flow.shape[0]
		cert_mat = np.ones((y_dim, x_dim))

		# prewitt kernel - works with greyscale images, like the optical flow algorithm 
		x_kernel = [[-0.5, -0.5, -0.5],[0., 0., 0.],[0.5, 0.5, 0.5]]
		x_kernel = np.array(x_kernel, np.float32)
		y_kernel = [[-0.5, 0., 0.5],[-0.5, 0., 0.5],[-0.5, 0., 0.5]]
		y_kernel = np.array(y_kernel, np.float32)
	
		flow_x_dx = cv2.filter2D(flow[:,:,0],-1,x_kernel)
		flow_x_dy = cv2.filter2D(flow[:,:,0],-1,y_kernel)
		dx = np.stack((flow_x_dx, flow_x_dy), axis = -1)

		flow_y_dx = cv2.filter2D(flow[:,:,1],-1,x_kernel)
		flow_y_dy = cv2.filter2D(flow[:,:,1],-1,y_kernel)
		dy = np.stack((flow_y_dx, flow_y_dy), axis = -1)

		motion_edg = np.zeros((y_dim,x_dim))

		for i in range(y_dim):
			for j in range(x_dim): 
				motion_edg[i,j] = dy[i,j,0]*dy[i,j,0] + dy[i,j,1]*dy[i,j,1] + dx[i,j,0]*dx[i,j,0] + dx[i,j,1]*dx[i,j,1]

				if motion_edg[i,j] > thresh: 
					cert_mat[i, j] = 0.0

		cert_mat = np.clip(cert_mat, 0.0, 1.0)

		return cert_mat	

# https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Real-Time_Neural_Style_CVPR_2017_paper.pdf
# temporal loss - difference between stylised output at t and warped stylised output at t - 1
# x: stylised image at time t 
# w: warped stylised image at time t-1
# c: flow weights (get flow weights)
def temporal_loss(x, w, c):
	x = tf.image.rgb_to_grayscale(x)
	c = c[np.newaxis,:,:]
	w = w[np.newaxis,:,:]
	D = 224 * 224 
	# difference between stylised frame and warped stylised frame 
	# mulitply by c - losses in occluded areas added
	loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
	loss = tf.cast(loss, tf.float32)
	return loss

def conv_flow_rgb(flow):
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	return rgb

def unprocess_out(out):
	im = Image.fromarray(np.uint8(out[0]))
	im = np.array(im)
	return im

def conv_flow_img(count, hsv, for_flow, back_flow):
	f_rgb = conv_flow_rgb(for_flow)
	b_rgb = conv_flow_rgb(back_flow)
	cv2.imshow('f_rgb image',f_rgb)
	cv2.waitKey(0)
	cv2.imwrite("./input_flow/test2/" + str(count) + ".jpg", f_rgb)
	cv2.imwrite("./input_flow/bo/backward/" + str(c) + ".jpg", b_rgb)
	count += 1

with tf.device('/gpu:0'):

	input = tf.placeholder(tf.float32, shape=[b_size, 224, 224, 3], name='input')
	# initialise net
	trans_net = transformer(input, norm)
	saver = tf.train.Saver(restore_sequentially=True)

	style_img = tf.placeholder(tf.float32, shape=[b_size, 224, 224, 3], name="style_img")

	output = trans_net(input, norm)

	vgg_style = vgg19(style_img)
	vgg_content = vgg19(input)
	vgg_outp = vgg19(output)
	
	style_outputs = [gram_matrix(vgg_style.__dict__[style_output]) for style_output in style_layers]
	style_targets = [gram_matrix(vgg_outp.__dict__[style_output]) for style_output in style_layers]

	content_outputs = [(vgg_content.__dict__[content_output]) for content_output in content_layers]
	content_targets = [(vgg_outp.__dict__[content_output]) for content_output in content_layers]

	style_loss = tf.zeros(b_size, tf.float32)
	for i in range(len(style_outputs)):
			style_loss += style_weight * tf.reduce_mean(tf.subtract(style_targets[i], style_outputs[i]) ** 2, [1, 2])
	
	content_loss = tf.zeros(b_size, tf.float32)
	for i in range(len(content_layers)):
		content_loss += content_weight * tf.reduce_mean(tf.subtract(content_targets[i], content_outputs[i]) ** 2, [1, 2, 3])

	# variation of pixels within an image 
	var_loss = var_weight * tf.image.total_variation(output)

	temp_loss = tf.zeros(b_size, tf.float32)

	temp_c = tf.placeholder(tf.float32, shape=[224,224], name='temp_c')
	temp_w = tf.placeholder(tf.float32, shape=[224,224], name='temp_w')

	if (tf.math.count_nonzero(temp_c) != 0):
		temp_loss = temporal_weight * temporal_loss(output, temp_w, temp_c)

	total_loss = style_loss + content_loss + var_loss + temp_loss

	train = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

	sty = args.style

	ckpt_directory = './checkpts/{}/'.format(sty)
	if not os.path.exists(ckpt_directory):
		os.makedirs(ckpt_directory)

	tf.global_variables_initializer().run()	

	style = get_style_img(sty)
	# dict must be array elements 
	style_np = [style for x in range(b_size)]

	# gets all content images you need - video frames
	imgs = get_train_imgs('frame')
	print('img length: {}'.format(len(imgs)))

	iter = int(num_data / b_size) 

	t0 = time.time()

	for e in range(epoch):
		inp_imgs = np.zeros((b_size, 224, 224, 3), dtype=np.float32)
		for i in range(iter):
			for j in range(b_size):
				im = imgs[i * b_size + j]
				inp_imgs[j] = preprocess_img(Image.open(im).convert('RGB'))

			if (i > 0):
				# shape - actual image shape 
				for_flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev_im[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(inp_imgs[0], cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
				back_flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(inp_imgs[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_im[0], cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)

				# certainty, comparing forward and backward flow 
				c = 1.0 - get_flow_weights_bounds(for_flow, 0.1)
				w = warp_flow(prev_im_stylised, back_flow)
				w = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)

			else:
				c = np.zeros((224, 224))
				w = np.zeros((224, 224))

			dict = {input: inp_imgs, style_img: style_np, temp_c:c, temp_w:w}

			loss, out, t_loss, s_loss, _ = sess.run([total_loss, output, temp_loss, style_loss, train], feed_dict=dict)

			prev_im = inp_imgs.copy()
			prev_im_stylised = unprocess_out(out.copy())

			print('iter {}/{} loss: {}'.format(i + 1, iter, loss[0]))

			if (i*j + i % 1000 == 0):
				input_shape = np.array(Image.open(im).convert('RGB')).shape
				out_im = unprocess_img(out, input_shape)
				cv2.imwrite(progress + sty + "_" + str(e) + "_" + str(i) + ".jpg", out_im)
				cv2.imwrite(progress + sty + "_content_" + str(e) + "_" + str(i) + ".jpg", inp_imgs[0])
			if (i*j + i % 1 == 0):
				f = open("./test_output/trainflow_loss.txt", "a")
				f.write(str(loss[0]) + ' ') 
				f.close()

	t1 = time.time()
	saver.save(sess, ckpt_directory + sty, global_step=e)
	total_time = t1-t0
	f = open("./test_output/trainflow_loss.txt", "a")
	f.write('\ntime: ' + str(total_time) + 'seconds')  
	f.close()