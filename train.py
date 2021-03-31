# TRAINS TRANSFORMER NET 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from transformer import transformer
from vgg19 import vgg19
import cv2
import numpy as np
from PIL import Image
import time
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from argparse import ArgumentParser

print("Opening Tensorflow with CUDA and CUDNN")

parser = ArgumentParser()
parser.add_argument('--style', '-s', type=str)
parser.add_argument('--b_size', '-b', type=int, default=1)
parser.add_argument('--norm', '-n', type=str, default="b")
args = parser.parse_args()

b_size = args.b_size
sty = args.style
norm = args.norm 

style_layers = ['conv1_1',
								'conv2_1',
								'conv3_1', 
								'conv4_1', 
								'conv5_1']
content_layers = ['conv4_2']

epoch = 2
style_weight = 1e0
content_weight = 1e0
learn_rate = 1e-3
var_weight = 10e-4

tr = './training_dataset_4/'
list = os.listdir('./training_dataset_4')
num_data = len(list)

progress = './test_output/progress/'

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

def conv_col(im):
	b, g, r = im.split()
	im = Image.merge("RGB", (r, g, b))
	return im

with tf.device('/gpu:0'):

	input = tf.placeholder(tf.float32, shape=[b_size, 224, 224, 3], name='input')
	# initialise net
	trans_net = transformer(input, norm)
	saver = tf.train.Saver(restore_sequentially=True)

	style_img = tf.placeholder(tf.float32, shape=[b_size, 224, 224, 3])

	# original non padded output, sliced padded output 
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

	var_loss = var_weight * tf.image.total_variation(output)

	total_loss = style_loss + content_loss + var_loss

	train = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

	ckpt_directory = './checkpts/{}/'.format(sty)
	if not os.path.exists(ckpt_directory):
		os.makedirs(ckpt_directory)

	tf.global_variables_initializer().run()	

	style = get_style_img(sty)
	# dict must be array elements 
	style_np = [style for x in range(b_size)]

	# gets all content images you need - video frames
	imgs = get_train_imgs('frame')
	print('Number of training images: {}'.format(len(imgs)))

	iter = int(num_data / b_size)

	t0 = time.time()
	print("Begin Training Network with style", sty)

	for e in range(epoch):
		inp_imgs = np.zeros((b_size, 224, 224, 3), dtype=np.float32)
		for i in range(iter):
			for j in range(b_size):
				im = imgs[i * b_size + j]
				inp_imgs[j] = preprocess_img(Image.open(im).convert('RGB'))
			dict = {input: inp_imgs, style_img: style_np}
			loss, out, _ = sess.run([total_loss, output, train], feed_dict=dict)
			print('iter {}/{} loss: {}'.format(i + 1, iter, loss[0]))

			if (i*j + i % 1000 == 0):
				input_shape = np.array(Image.open(im).convert('RGB')).shape
				out_im = unprocess_img(out, input_shape)
				cv2.imwrite(progress + sty + "_" + str(e) + "_" + str(i) + ".jpg", out_im)
				cv2.imwrite(progress + sty + "_content_" + str(e) + "_" + str(i) + ".jpg", np.array(conv_col(Image.open(im))))
			if (i*j + i % 1000 == 0):
				f = open("./test_output/train_loss.txt", "a")
				f.write(str(loss[0]) + ' ') 
				f.close()

	t1 = time.time()
	print("Saving Model")
	saver.save(sess, ckpt_directory + sty, global_step=e)
	total_time = t1-t0
	f = open("./test_output/train_loss.txt", "a")
	f.write('\ntime: ' + str(total_time) + ' seconds')  
	f.close()
