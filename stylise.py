import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser

import vgg19 

from variables import b_size, epoch, sty, cont

input_dir = './input_images/'
style_dir = './style_images/'

parser = ArgumentParser()
parser.add_argument('--mode', '-m', type=str)
args = parser.parse_args()
mode = args.mode

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

def create_vid(img, video, input_shape):
	img = img[0]
	# print(img)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	im = Image.fromarray(np.uint8(img))
	# resample NEAREST, BILINEAR, BICUBIC, ANTIALIAS 
	# filters for when resizing, change num pixels rather than resize 
	im = im.resize((input_shape[1], input_shape[0]), resample=Image.LANCZOS)
	# print("written image has size {}, {}".format(input_shape[1], input_shape[0]))
	im = np.array(im)

	video.write(im)

def stylise(img, style):
	with tf.device('/gpu:0'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			input_checkpoint = './checkpts/{}/{}-{}'.format(style, style, epoch-1)
			saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
			saver.restore(sess, input_checkpoint)
			graph = tf.get_default_graph()

			#print(sess.run(graph.get_tensor_by_name(':0')))
						
			input_image_ten = graph.get_tensor_by_name('input:0')
			output_ten = graph.get_tensor_by_name('output:0')

			dir_name = './input_images/' + img
			dir_list = os.listdir(dir_name)

			first_img_w, first_img_h = Image.open(dir_name + '/' + dir_list[0]).size

			fourcc = cv2.VideoWriter_fourcc(*'DIVX')
			video = cv2.VideoWriter("./output_images/" + img + "_" + style + ".avi", fourcc, 17.0, (first_img_w, first_img_h))
			
			for frame in dir_list:
				n = frame.split(".")[0]
				input_img = Image.open(dir_name + '/' + frame).convert('RGB')

				input_shape = np.array(input_img).shape
				input_img = process_img(input_img)

				out = sess.run(output_ten, feed_dict={input_image_ten: input_img})

				create_vid(out, video, input_shape)

			video.release()
			cv2.destroyAllWindows()

def main():
	stylise(cont, sty)

if __name__ == "__main__":
		main()