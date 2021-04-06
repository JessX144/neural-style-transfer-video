import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import threading
from tkinter import ttk

window_title = "Stylise Video"

class MyVideoCapture:

	def __init__(self, video_source=0, width=None, height=None, fps=None):
	
		self.video_source = video_source
		self.width = width
		self.height = height
		self.fps = fps
		
		# Open the video source
		self.vid = cv2.VideoCapture(video_source)
		if not self.vid.isOpened():
			raise ValueError("[MyVideoCapture] Unable to open video source", video_source)

		# Get video source width and height
		if not self.width:
			self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))	# convert float to int
		if not self.height:
			self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int
		if not self.fps:
			self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))  # convert float to int

		# default value at start		
		self.ret = False
		self.frame = None
		
		self.convert_color = cv2.COLOR_BGR2RGB
		#self.convert_color = cv2.COLOR_BGR2GRAY
		self.convert_pillow = True
		
		# start thread
		self.running = True
		self.thread = threading.Thread(target=self.process)
		self.thread.start()
		
	def process(self):
		frame_counter = 0
		while self.running:
			ret, frame = self.vid.read()
			frame_counter += 1

			if frame_counter == self.vid.get(cv2.CAP_PROP_FRAME_COUNT):
				frame_counter = 0 #Or whatever as long as it is the same as next line
				self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

			if ret:
				# process image
				frame = cv2.resize(frame, (self.width, self.height))
					
				if self.convert_pillow:
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					frame = PIL.Image.fromarray(frame)

			# assign new frame
			self.ret = ret
			self.frame = frame

			# sleep for next frame
			time.sleep(1/self.fps)
		
	def get_frame(self):
		return self.ret, self.frame
	
	# Release the video source when the object is destroyed
	def __del__(self):
		# stop thread
		if self.running:
			self.running = False
			self.thread.join()

		# relase stream
		if self.vid.isOpened():
			self.vid.release()
			
 
class tkCamera(tkinter.Frame):

	def __init__(self, window, text="", video_source=0, width=None, height=None):
		super().__init__(window)
		
		self.window = window

		self.window.maxsize(900, 600)
		
		self.window.title(window_title)
		self.video_source = video_source
		self.vid = MyVideoCapture(self.video_source, width, height)

		self.label = tkinter.Label(self, text=text)
		self.label.pack()
		
		self.canvas = tkinter.Canvas(self, width=self.vid.width, height=self.vid.height)
		self.canvas.pack()
		 
		# After it is called once, the update method will be automatically called every delay milliseconds
		# calculate delay using `FPS`
		self.delay = int(1000/self.vid.fps)

		self.image = None
		
		self.running = True
		self.update_frame()
			
	def update_frame(self):
		# widgets in tkinter already have method `update()` so I have to use different name -

		# Get a frame from the video source
		ret, frame = self.vid.get_frame()
		
		if ret:
			#self.image = PIL.Image.fromarray(frame)
			self.image = frame
			self.photo = PIL.ImageTk.PhotoImage(image=self.image)
			self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
		
		if self.running:
			self.window.after(self.delay, self.update_frame)

class App:

	def __init__(self, window, window_title, video_sources):

		self.window = window

		self.window.title(window_title)
		
		self.vids = []

		columns = 3
		for number, source in enumerate(video_sources):
			text, stream = source
			vid = tkCamera(self.window, text, stream, 300, 300)
			x = number % columns
			y = number // columns
			vid.grid(row=y, column=x)
			self.vids.append(vid)
		
		self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
		self.window.mainloop()
	
	def on_closing(self, event=None):
		for source in self.vids:
			source.vid.running = False
		print('[App] exit')
		self.window.destroy()

if __name__ == '__main__':	 

	sources = [
		('Train 1', './output_images/bo_face_noise.avi'),
		('Train 2', './output_images/bo_face_noise_og.avi'),
		('Train 3', './output_images/bo_face_train.avi'),
		('Original Video', './output_images/bo_face_train_og.avi'),
		('Optical Flow', './test_output/bo_flow.avi'),
		('Confidence Matrix', './test_output/bo_conf.avi')
		]
		
	# Create a window and pass it to the Application object
	App(tkinter.Tk(), "Tkinter and OpenCV", sources)