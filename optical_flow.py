import cv2
import numpy as np

TAG_FLOAT = 202021.25  
MOTION_BOUNDARIE_VALUE = 0.0

content = "bo"
cap = cv2.VideoCapture("./input_images/" + content + ".avi")

# number of frames to consider 
prev_frames = 1

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter("./input_flow/" + content + ".avi", fourcc, 25.0, (frame1.shape[0], frame1.shape[1]))		

def get_flow_weights(flow1, flow2): 
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
				reliable[ay, ax] = MOTION_BOUNDARIE_VALUE
				continue
			

	print("reliable shape:")
	print(reliable.shape)

	#need to apply smoothing to reliable mat
	reliable = cv2.GaussianBlur(reliable,(3,3),0)
	reliable = np.clip(reliable, 0.0, 255.0)		
	return reliable		

def flow_to_img(flow, normalize=True, info=None, flow_mag_max=None):

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add text to the image, if requested
    if info is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, info, (20, 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return img

def flow_write(flow, dst_file):

    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)

def read_flow(rgb):
		flow = rgb[:, :, 2:0:-1].astype(np.float32)
		flow = flow - 32768
		flow = flow / 64

		# Clip flow values
		flow[np.abs(flow) < 1e-10] = 1e-10

		# Remove invalid flow values
		invalid = (rgb[:, :, 0] == 0)
		flow[invalid, :] = 0
		return flow

count = 0;

while(cap.isOpened()):
		ret, frame2 = cap.read()
		print("shape:")
		print(frame2.shape)
		if not ret:
			break

		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		flow_b = cv2.calcOpticalFlowFarneback(next,prvs, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		print("flow shapes:")
		print(flow.shape)

		get_flow_weights(flow, flow_b)

		print("imgs:")
		print(prvs.shape)
		print("imgs2:")
		print(next.shape)

		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

		video.write(rgb)
		prvs = next

		fl = read_flow(rgb)
		flow_write(fl, "./input_flow/test.flo")
		img = flow_to_img(fl)
		cv2.imwrite("./input_flow/test/" + str(count) + ".jpg", img)
		count += 1

video.release()
cap.release()
cv2.destroyAllWindows()