import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

import vgg19 
from transformer import transformer

# To avoid having to train the model, we take pretrained weights 
model_weights = 'vgg19.npy'

input_dir = './input_images/'
style_dir = './style_images/'

def get_img(img, img_dir):
	img = cv2.imread(img_dir + str(img) + '.jpg', cv2.IMREAD_COLOR)
	return img

def write_img(img_name, img):
	cv2.imwrite('./output_images/' + img_name + '_output.jpg', img)
	return img

def process_img(img):
	return img

def stylise(img, style):
  with tf.device('/gpu:0'):

    style_img = get_img(style, style_dir)
    input_image = get_img(img, input_dir)

    img = np.zeros((1, 224, 224, 3), dtype=np.float32)

    # first element in batch 
    img[0] = np.asarray(Image.fromarray(input_image).convert('RGB').resize((224, 224)), np.float32)
    vgg19.vgg19(img)  

    #write_img(img, input_img)
    #write_img(style, style_img)
    transformer(img)
    return img

def main():
  stylise("cloud","lions")

if __name__ == "__main__":
    main()