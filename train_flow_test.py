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
import math 

from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, mean_squared_error

tr = './testing_flows/alley_1_im/'
list = os.listdir('./testing_flows/alley_1_im') # dir is your directory path
num_data = len(list)

hsv = np.zeros((224,224,3))
hsv[...,1] = 255

def preprocess_img(img):

	imgpre = img.copy()
	imgpre = imgpre.resize((224, 224))
	imgpre = np.asarray(imgpre, dtype=np.float32)
	
	return imgpre

def get_train_imgs(name):
	imgs = []
	# cannot give all imgs, memory error, use list 
	for filename in list:
		if (name in os.path.splitext(filename)[0]):
			imgs.append(tr + filename)
	return imgs

# warps an image according to its flow, predicts its movement 
# from OpenCV
def warp_flow(img, flow):
		h, w = flow.shape[:2]
		flow = -flow
		flow[:,:,0] += np.arange(w)
		flow[:,:,1] += np.arange(h)[:,np.newaxis]
		res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
		return res

# flow weights by comparing motion boundaries 
def get_flow_weights_bounds(flow, thresh): 

		xSize = flow.shape[1]
		ySize = flow.shape[0]
		reliable = np.ones((ySize, xSize))

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

		motionEdge = np.zeros((ySize,xSize))

		for i in range(ySize):
			for j in range(xSize): 
				motionEdge[i,j] = dy[i,j,0]*dy[i,j,0] + dy[i,j,1]*dy[i,j,1] + dx[i,j,0]*dx[i,j,0] + dx[i,j,1]*dx[i,j,1]

				if motionEdge[i,j] > thresh: 
					reliable[i, j] = 0.0

		reliable = np.clip(reliable, 0.0, 1.0)

		return reliable	

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
	# mulitply by c - losses in occluded areas not taken 
	loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
	loss = tf.cast(loss, tf.float32)
	return loss

def conv_flow_rgb(flow):
	print(flow.shape)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	hsv_i = np.uint8(hsv)
	rgb = cv2.cvtColor(hsv_i,cv2.COLOR_HSV2BGR)
	return rgb

def _color_wheel():
    # Original inspiration: http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # RGB

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    # YG
    colorwheel[col: YG + col, 0] = 255 - \
        np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col: YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col: GC + col, 1] = 255
    colorwheel[col: GC + col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC

    # CB
    colorwheel[col: CB + col, 1] = 255 - \
        np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col: CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col: BM + col, 2] = 255
    colorwheel[col: BM + col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM

    # MR
    colorwheel[col: MR + col, 2] = 255 - \
        np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col: MR + col, 0] = 255

    return colorwheel

def _normalize_flow(flow):
    UNKNOWN_FLOW_THRESH = 1e9
    # UNKNOWN_FLOW = 1e10

    height, width, nBands = flow.shape
    if not nBands == 2:
        raise AssertionError("Image must have two bands. [{h},{w},{nb}] shape given instead".format(
            h=height, w=width, nb=nBands))

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Fix unknown flow
    idxUnknown = np.where(np.logical_or(
        abs(u) > UNKNOWN_FLOW_THRESH,
        abs(v) > UNKNOWN_FLOW_THRESH
    ))
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max([-999, np.max(u)])
    maxv = max([-999, np.max(v)])
    minu = max([999, np.min(u)])
    minv = max([999, np.min(v)])

    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([-1, np.max(rad)])

    eps = np.finfo(np.float32).eps
    u = u/(maxrad + eps)
    v = v/(maxrad + eps)

    return u, v

def _compute_color(u, v):
    colorwheel = _color_wheel()
    idxNans = np.where(np.logical_or(
        np.isnan(u),
        np.isnan(v)
    ))
    u[idxNans] = 0
    v[idxNans] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1)
    k0 = fk.astype(np.uint8)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]

    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1-f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)  # RGB
        # img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8) # BGR

    return img.astype(np.uint8)

def read_flow(flo):
	tag = np.fromfile(flo, np.float32, count=1)[0]
	width = np.fromfile(flo, np.int32, count=1)[0]
	height = np.fromfile(flo, np.int32, count=1)[0]

	nbands = 2
	tmp = np.frombuffer(flo.read(nbands * width * height * 4),
											np.float32, count=nbands * width * height)
	flow = np.resize(tmp, (int(height), int(width), int(nbands)))
	flo.close()
	return flow

def flo_visualise(flo):
	flow = read_flow(flo)
	im = _flow2color(flow)
	return im, flow

def _flow2color(flow):
    u, v = _normalize_flow(flow)
    img = _compute_color(u, v)

    return img

def rmse(predictions, targets):
		x = (targets - predictions) 
		x_sq = np.square(x)	
		x_sq_sum = x_sq.sum()
		x_mean = x_sq_sum / (224*224*2)
		x_sqrt = math.sqrt(x_mean)
		return x_sqrt

opt_flow_loss = []
certainty_flow_loss = []

# gets all content images you need - video frames
imgs = get_train_imgs('frame')
print('img length: {}'.format(len(imgs)))

inp_imgs = np.zeros((224, 224, 3), dtype=np.float32)
# we evaluate pairs of frames
for i in range(num_data):
	im = imgs[i]
	im_next = imgs[i + 1]
	inp_imgs = np.array(Image.open(im_next).resize((224, 224)).convert('RGB'))
	prev_im = np.array(Image.open(im).resize((224, 224)).convert('RGB'))

	for_flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev_im, cv2.COLOR_BGR2GRAY), cv2.cvtColor(inp_imgs, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flow_im = _flow2color(for_flow)

	ground_flow_for = open('./testing_flows/alley_1_flow/frame_' + str(i+1).zfill(4) + '.flo', 'rb')
	ground_flow_for = read_flow(ground_flow_for)
	ground_flow_for = cv2.resize(ground_flow_for, dsize=(224, 224))
	g_flow_im_for = _flow2color(ground_flow_for)

	cv2.imshow('ground forward flow', g_flow_im_for)
	cv2.waitKey(0)
	cv2.imshow('forward flow', flow_im)
	cv2.waitKey(0)

	opt_loss_for = rmse(for_flow, ground_flow_for)
	print("rmse, range:", opt_loss_for, max(np.ptp(for_flow), np.ptp(ground_flow_for)))
	print("rmse percentage:", opt_loss_for/max(np.ptp(for_flow), np.ptp(ground_flow_for)))

	occ_im = cv2.imread('./testing_flows/alley_1_occ/frame_' + str(i+1).zfill(4) + '.png')
	occ_im = cv2.cvtColor(occ_im, cv2.COLOR_BGR2GRAY)
	occ_im = Image.fromarray(occ_im).resize((224, 224))
	occ_im = np.array(occ_im).astype(np.float32)
	occ_im = np.clip(occ_im, 0.0, 1.0)

	# certainty, comparing forward and backward flow 
	c = 1.0 - get_flow_weights_bounds(for_flow, 0.1)

	cv2.imshow("c", c)
	cv2.waitKey(0)
	cv2.imshow("occ_im", occ_im)
	cv2.waitKey(0)

	f_loss = f1_score(occ_im.ravel(), c.ravel(), average='macro')

	certainty_flow_loss.append(f_loss)

	#tn, fp, fn, tp = confusion_matrix(occ_im.ravel(), c.ravel()).ravel()
	#acc = (tp+tn)/(224*224)
	#sensitivity = tp/(tp+fn)
	#specificity = tn/(tn+fp)
	#certainty_sens_loss.append(sensitivity)
	#certainty_spec_loss.append(specificity)
		
	print('iter {}/{}'.format(i, num_data))

avg_opt_flow_loss = np.average(opt_flow_loss)
avg_certainty_loss = np.average(certainty_flow_loss)
print("average optical flow loss against ground value: ", avg_opt_flow_loss)
print("average certainity f-measure against ground value: ", avg_certainty_loss)
