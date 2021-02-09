import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import vgg19 
from transformer import transformer

from variables import b_size, epoch

input_dir = './input_images/'
style_dir = './style_images/'

def get_img(img, img_dir):
	img = Image.open(img_dir + str(img) + '.jpg').convert('RGB')
	return img

def write_img(img_name, img):
	cv2.imwrite('./output_images/' + img_name + '_output.jpg', img)
	return img

def process_img(img):
	img = np.asarray(img.resize((224, 224)), dtype=np.float32)
	arrays = [img for _ in range(b_size)]
	inp_img = np.stack(arrays, axis=0)
	return inp_img

# reverse processing 
def unprocess_img(img, style_name, input_name, input_shape):
	#print("img.shape:")
	#print(img.shape)
	img = img[0]
	im = Image.fromarray(np.uint8(img))
	im = im.resize((input_shape[0], input_shape[1]))
	write_img('{}_{}'.format(input_name, style_name), img)

def stylise(img, style):
	with tf.device('/gpu:0'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			input_img = get_img(img, input_dir)
			input_shape = np.array(input_img).shape
			input_img = process_img(input_img)

			input_checkpoint = './checkpts/{}/{}-{}'.format(style, style, epoch)
			saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
			saver.restore(sess, input_checkpoint)
			graph = tf.get_default_graph()

			tf.global_variables_initializer().run()	
			tf.local_variables_initializer().run()	

			#print(sess.run(graph.get_tensor_by_name(':0')))
						
			input_image_ten = graph.get_tensor_by_name('input:0')
			output_ten = graph.get_tensor_by_name('output:0')
			#output_ten1 = graph.get_tensor_by_name('output1:0')
			#output_ten2 = graph.get_tensor_by_name('output2:0')
			#output_ten3 = graph.get_tensor_by_name('output3:0')

			out = sess.run(output_ten, feed_dict={input_image_ten: input_img})
			#out1 = sess.run(output_ten1, feed_dict={input_image_ten: input_img})
			#out2 = sess.run(output_ten2, feed_dict={input_image_ten: input_img})
			#out3 = sess.run(output_ten3, feed_dict={input_image_ten: input_img})

			#print(out1)
			#print(out2)
			#print(out3)

			unprocess_img(out, style, img, input_shape)

def main():
	stylise("jake","lions")

if __name__ == "__main__":
		main()