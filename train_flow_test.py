# TRAINS TRANSFORMER NET - takes into account optical flow 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformer import transformer
from vgg19 import vgg19
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
import time
from argparse import ArgumentParser
import math 
import time
from tkinter import Tk, Canvas, NW, mainloop, Label

from sklearn.metrics import f1_score

dir = "alley_1"

time_ = []
time_rud = []

tr = './testing_flows/' + dir + '_im/'
list = os.listdir('./testing_flows/'+ dir + '_im') # dir is your directory path
flow_f = './testing_flows/' + dir +'_flow/frame_'
occ_f = './testing_flows/'+ dir + '_occ/frame_'
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

# Confidence matrix calculations for Ruder implementation 
# https://github.com/coyotestarrkwsq/style-transfer-for-video/blob/3dc32066af1292fbbc9e3977ba9a54652a6b14a2/flow.py
def get_flow_weights_bound_ruder(flow1, flow2): 

	a_time = time.time()

	xSize = flow1.shape[1]
	ySize = flow1.shape[0]
	reliable = 255 * np.ones((ySize, xSize))

	size = xSize * ySize

	x_kernel = [[-0.5, -0.5, -0.5],[0., 0., 0.],[0.5, 0.5, 0.5]]
	x_kernel = np.array(x_kernel, np.float32)
	y_kernel = [[-0.5, 0., 0.5],[-0.5, 0., 0.5],[-0.5, 0., 0.5]]
	y_kernel = np.array(y_kernel, np.float32)
	
	flow_x_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
	flow_x_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
	dx = np.stack((flow_x_dx, flow_x_dy), axis = -1)

	flow_y_dx = cv2.filter2D(flow1[:,:,0],-1,x_kernel)
	flow_y_dy = cv2.filter2D(flow1[:,:,0],-1,y_kernel)
	dy = np.stack((flow_y_dx, flow_y_dy), axis = -1)

	motionEdge = np.zeros((ySize,xSize))

	for i in range(ySize):
		for j in range(xSize): 
			motionEdge[i,j] += dy[i,j,0]*dy[i,j,0]
			motionEdge[i,j] += dy[i,j,1]*dy[i,j,1]
			motionEdge[i,j] += dx[i,j,0]*dx[i,j,0]
			motionEdge[i,j] += dx[i,j,1]*dx[i,j,1]
			
	for ax in range(xSize):
		for ay in range(ySize): 
			bx = ax + flow1[ay, ax, 0]
			by = ay + flow1[ay, ax, 1]		

			x1 = int(bx)
			y1 = int(by)
			x2 = x1 + 1
			y2 = y1 + 1
			
			if x1 < 0 or x2 >= xSize or y1 < 0 or y2 >= ySize:
				reliable[ay, ax] = 0.0
				continue 
			
			alphaX = bx - x1 
			alphaY = by - y1

			a = (1.0-alphaX) * flow2[y1, x1, 0] + alphaX * flow2[y1, x2, 0]
			b = (1.0-alphaX) * flow2[y2, x1, 0] + alphaX * flow2[y2, x2, 0]
			
			u = (1.0 - alphaY) * a + alphaY * b
			
			a = (1.0-alphaX) * flow2[y1, x1, 1] + alphaX * flow2[y1, x2, 1]
			b = (1.0-alphaX) * flow2[y2, x1, 1] + alphaX * flow2[y2, x2, 1]
			
			v = (1.0 - alphaY) * a + alphaY * b
			cx = bx + u
			cy = by + v
			u2 = flow1[ay,ax,0]
			v2 = flow1[ay,ax,1]
			
			if ((cx-ax) * (cx-ax) + (cy-ay) * (cy-ay)) >= 0.01 * (u2*u2 + v2*v2 + u*u + v*v) + 0.5: 
				# Set to a negative value so that when smoothing is applied the smoothing goes "to the outside".
				# Afterwards, we clip values below 0.
				reliable[ay, ax] = -255.0
				continue
			
			if motionEdge[ay, ax] > 0.01 * (u2*u2 + v2*v2) + 0.002: 
				reliable[ay, ax] = 0.0
				continue
			
	#need to apply smoothing to reliable mat
	reliable = cv2.GaussianBlur(reliable,(3,3),0)
	ret, reliable = cv2.threshold(reliable,10,1.0,cv2.THRESH_BINARY)
	reliable = np.clip(reliable, 0.0, 1.0)	 

	b_time = time.time()
	delta = b_time - a_time
	time_rud.append(delta)
	return reliable	

# flow weights by comparing motion boundaries 
def get_flow_weights_bounds(flow, thresh): 

		a = time.time()

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

		b = time.time()
		delta = b - a
		time_.append(delta)
		return cert_mat	

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
	loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
	loss = tf.cast(loss, tf.float32)
	return loss

################################# OPENCV IMPLEMENTATION TO VISUALISE OPTICAL FLOW
def conv_flow_rgb(flow):
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	hsv_i = np.uint8(hsv)
	rgb = cv2.cvtColor(hsv_i,cv2.COLOR_HSV2BGR)
	return rgb

def _color_wheel():
		RY = 15
		YG = 6
		GC = 4
		CB = 11
		BM = 13
		MR = 6

		ncols = RY + YG + GC + CB + BM + MR

		colorwheel = np.zeros([ncols, 3])	# RGB

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
				img[:, :, i] = np.floor(255 * col).astype(np.uint8)	

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
################################# 

def rmse(predictions, targets):
		x = (targets - predictions) 
		x_sq = np.square(x)	
		x_sq_sum = x_sq.sum()
		x_mean = x_sq_sum / (224*224*2)
		x_sqrt = math.sqrt(x_mean)
		return x_sqrt

certainty_flow_loss = []
certainty_flow_loss_rud = []

rmspe_flow_loss = []
rmspe_flow_loss_rud = []

def get_loss():

	# gets all content images you need - video frames
	imgs = get_train_imgs('frame')
	print('img length: {}'.format(len(imgs)))

	inp_imgs = np.zeros((224, 224, 3), dtype=np.float32)
	# we evaluate pairs of frames
	for i in range(num_data-1):
		root = Tk()	
		im = imgs[i]
		im_next = imgs[i + 1]
		inp_imgs = np.array(Image.open(im_next).resize((224, 224)).convert('RGB'))
		prev_im = np.array(Image.open(im).resize((224, 224)).convert('RGB'))

		for_flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev_im, cv2.COLOR_BGR2GRAY), cv2.cvtColor(inp_imgs, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
		back_flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(inp_imgs, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_im, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
		flow_im = _flow2color(for_flow)

		ground_flow_for = open(flow_f + str(i+1).zfill(4) + '.flo', 'rb')
		ground_flow_for = read_flow(ground_flow_for)
		ground_flow_for = cv2.resize(ground_flow_for, dsize=(224, 224))
		g_flow_im_for = _flow2color(ground_flow_for)

		g_flow_im_for = cv2.cvtColor(g_flow_im_for, cv2.COLOR_BGR2RGB)
		tk_g_flow_im_for = ImageTk.PhotoImage(image=Image.fromarray(g_flow_im_for))		
		label = Label(root, image = tk_g_flow_im_for)
		label.image = tk_g_flow_im_for
		label.grid(row=0, column=0, columnspan=3)

		flow_im = cv2.cvtColor(flow_im, cv2.COLOR_BGR2RGB)
		tk_flow_im = ImageTk.PhotoImage(image=Image.fromarray(flow_im))		
		label2 = Label(root, image = tk_flow_im)
		label2.image = tk_flow_im
		label2.grid(row=0, column=1, columnspan=3)

		opt_loss_for = rmse(for_flow, ground_flow_for)

		occ_im = cv2.imread(occ_f + str(i+1).zfill(4) + '.png')
		occ_im = cv2.cvtColor(occ_im, cv2.COLOR_BGR2GRAY)
		occ_im = Image.fromarray(occ_im).resize((224, 224))
		occ_im = np.array(occ_im).astype(np.float32)
		occ_im = np.clip(occ_im, 0.0, 1.0)

		# certainty matrix of forward flow
		c = 1.0 - get_flow_weights_bounds(for_flow, 0.15)
		c_rud = 1.0 - get_flow_weights_bound_ruder(back_flow, for_flow)

		tk_c = ImageTk.PhotoImage(image=Image.fromarray(c*255))		
		label3 = Label(root, image = tk_c)
		label3.image = tk_c
		label3.grid(row=1, column=0)

		tk_c_rud = ImageTk.PhotoImage(image=Image.fromarray(c_rud*255))		
		label4 = Label(root, image = tk_c_rud)
		label4.image = tk_c_rud
		label4.grid(row=1, column=1)

		tk_occ_im = ImageTk.PhotoImage(image=Image.fromarray(occ_im*255))		
		label5 = Label(root, image = tk_occ_im)
		label5.image = tk_c_rud
		label5.grid(row=1, column=3)
 
		mainloop() 

		f_loss = f1_score(occ_im.ravel(), c.ravel(), average='macro')

		f_loss_r = f1_score(occ_im.ravel(), c_rud.ravel(), average='macro')

		f_loss_rmse = rmse(occ_im.ravel(), c.ravel())
		f_loss_r_rmse = rmse(occ_im.ravel(), c_rud.ravel())

		certainty_flow_loss.append(f_loss)
		certainty_flow_loss_rud.append(f_loss_r)

		rmspe_flow_loss.append(f_loss_rmse)
		rmspe_flow_loss_rud.append(f_loss_r_rmse)

		print('test image {}/{}'.format(i, num_data))

	avg_certainty_flow_loss = np.average(certainty_flow_loss)
	avg_certainty_flow_loss_rud = np.average(certainty_flow_loss_rud)

	avg_rmspe_flow_loss = np.average(rmspe_flow_loss)
	avg_rmspe_flow_loss_rud = np.average(rmspe_flow_loss_rud)

	print("average certainity f-measure against ground value: ", avg_certainty_flow_loss)
	print("average certainity f-measure against ground value ruder: ", avg_certainty_flow_loss_rud)

	print("average rmspe loss against ground value: ", avg_rmspe_flow_loss)
	print("average rmspe ruder against ground value: ", avg_rmspe_flow_loss_rud)

	print("average time: ", np.average(time_))
	print("average ruder time:", np.average(time_rud))

def main():
	get_loss()

if __name__ == "__main__":
		main()