from __future__ import unicode_literals
import numpy as np
import cv2
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from argparse import ArgumentParser
import vgg19 
import youtube_dl
import time

from train_flow_test import _flow2color, get_flow_weights_bounds

print("Opening Tensorflow with CUDA and CUDNN")

ydl_opts = {}

input_dir = './input_images/'
style_dir = './style_images/'

parser = ArgumentParser()
parser.add_argument('--style', '-s', type=str)
parser.add_argument('--content', '-c', type=str, default="e")
parser.add_argument('--batch', '-b', type=int, default=1)
parser.add_argument('--url', '-u', type=str, default="e")
parser.add_argument('--name', '-n', type=str, default="video")
parser.add_argument('--method', '-m', type=str, default="train")
args = parser.parse_args()

sty = args.style
cont = args.content
b_size = args.batch
url = args.url
name = args.name
method = args.method

epoch = 2

def get_img(img, img_dir):
	img = Image.open(img_dir + str(img) + '.jpg').convert('RGB')
	return img

def write_img(img_name, style_name, img):
	img.save('./output_images/{}_{}.jpg'.format(img_name, style_name))
	return img

def process_img(img):
	img = np.asarray(img.resize((224, 224)), dtype=np.float32)
	arrays = [img for _ in range(b_size)]
	inp_img = np.stack(arrays, axis=0)
	return inp_img

# reverse processing 
def unprocess_img(img, style_name, input_name, input_shape):
	img = img[0]

	im = Image.fromarray(np.uint8(img))
	# resample NEAREST, BILINEAR, BICUBIC, ANTIALIAS 
	# filters for when resizing, change num pixels rather than resize 
	im = im.resize((input_shape[1], input_shape[0]), resample=Image.LANCZOS)

	write_img(input_name, style_name, im)

def play_video(name):
	cap = cv2.VideoCapture(name) 
	
	# Check if camera opened successfully 
	if (cap.isOpened()== False):  
		print("Error opening video file") 
	
	# Read until video is completed 
	while(cap.isOpened()): 
		# Capture frame-by-frame 
		ret, frame = cap.read() 
		if ret == True: 

			# Display the resulting frame 
			cv2.imshow('Frame', frame) 
	
			# Press Q on keyboard to  exit 
			if cv2.waitKey(25) & 0xFF == ord('q'): 
				break
	
		# Break the loop 
		else:  
			break
	
	# When everything done, release  
	# the video capture object 
	cap.release() 
	
	# Closes all the frames 
	cv2.destroyAllWindows() 

def create_vid(img, video, input_shape):

	img = img[0]
	# remove padding
	img = img[10:-10,10:-10,:]

	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	im = Image.fromarray(np.uint8(img))
	# resample NEAREST, BILINEAR, BICUBIC, ANTIALIAS 
	# filters for when resizing, change num pixels rather than resize 
	im = im.resize((input_shape[1], input_shape[0]), resample=Image.BILINEAR)
	im = np.array(im)

	video.write(im)

from random import seed
from random import random
# seed random number generator
seed(1)

def generate_opt_vid(prev_img, curr_img, fl_vid, conf_vid):

	p_img = np.array(prev_img)
	c_img = np.array(curr_img)

	r = 224 / p_img.shape[0]

	dim = (224, int(p_img.shape[1] * r))
	p_img = cv2.resize(p_img, dim)
	c_img = cv2.resize(c_img, dim)

	for_flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
	fl_im = _flow2color(for_flow)
	c = 1.0 - get_flow_weights_bounds(for_flow, 0.1)

	#t1 = time.time()
	#print("time: ", t1-t0)

	conf_vid.write(c.astype(np.uint8) * 255)
	fl_vid.write(fl_im)

def write_frames(name):
	prefixed = [filename for filename in os.listdir('./input_images/') if name in filename]
	print(prefixed)
	filename, file_extension = os.path.splitext(prefixed[0])
	# print('./input_images/' + name + file_extension)
	vidcap = cv2.VideoCapture('./input_images/' + name + file_extension)

	success,image = vidcap.read()
	os.mkdir('./input_images/' + name)
	count = 0
	while success:
		cv2.imwrite('./input_images/' + name + '/' + "{:06d}".format(count) + '.jpg', image)     # save frame as JPEG file      
		success,image = vidcap.read()
		count += 1

def stylise(img, style):
	with tf.device('/gpu:0'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			input_checkpoint = './checkpts_' + method + '/{}/{}-{}'.format(style, style, epoch-1)
			saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
			saver.restore(sess, input_checkpoint)
			graph = tf.get_default_graph()

			input_image_ten = graph.get_tensor_by_name('input:0')
			output_ten = graph.get_tensor_by_name('output:0')

			# if a url has been provided 
			if not (url == "e"):
				with youtube_dl.YoutubeDL({'outtmpl': './input_images/' + name + '.%(ext)s'}) as ydl:
					#info_dict = ydl.extract_info(url, download=False)
					#first_img_w = info_dict.get("width", None)
					#first_img_h = info_dict.get("height", None)
					ydl.download([url])
					write_frames(name)
					img = name
					#dir_name = './input_images/' + img
					#dir_list = os.listdir(dir_name)
			elif not (img == "e"):
				# dir_name = './input_images/' + img
				if not os.path.exists('./input_images/' + img):
					write_frames(img)
				#dir_list = os.listdir(dir_name)
				#first_img_w, first_img_h = Image.open(dir_name + '/' + dir_list[0]).size

			dir_name = './input_images/' + img
			dir_list = os.listdir(dir_name)
			first_img_w, first_img_h = Image.open(dir_name + '/' + dir_list[0]).size

			fourcc = cv2.VideoWriter_fourcc(*'DIVX')
			video = cv2.VideoWriter("./output_images/" + img + "_" + style + "_" + method + ".avi", fourcc, 17.0, (first_img_w, first_img_h))

			scaled_h = int((first_img_h*224)/first_img_w)
			c_vid = cv2.VideoWriter("./test_output/" + img + "_" +  "conf.avi", fourcc, 17.0, (224, scaled_h), 0)
			flow_vid = cv2.VideoWriter("./test_output/" + img + "_" + "flow.avi", fourcc, 17.0, (224, scaled_h))

			t0 = time.time()
			print("Begin Stylising with style", sty)

			width, height = Image.open(dir_name + '/' + dir_list[0]).convert('RGB').size
			prev_image = Image.new('RGB', (width, height))

			for frame in dir_list:
				print("frame: ", frame)
				n = frame.split(".")[0]
				input_img = Image.open(dir_name + '/' + frame).convert('RGB')
				
				generate_opt_vid(input_img, prev_image, flow_vid, c_vid)
				prev_image = input_img

				input_shape = np.array(input_img).shape
				input_img = process_img(input_img)
				
				out = sess.run(output_ten, feed_dict={input_image_ten: input_img})

				create_vid(out, video, input_shape)
			
			print("Saving stylised video")

			t1 = time.time()
			video.release()
			c_vid.release()
			flow_vid.release()
			cv2.destroyAllWindows()
		
			total_time = t1-t0
			print("time to stylise: ", total_time, "seconds")

def main():

	stylise(cont, sty)

if __name__ == "__main__":
		main()