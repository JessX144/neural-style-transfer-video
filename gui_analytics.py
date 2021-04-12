import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import threading
from tkinter import ttk
import sys

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
		while(self.running):
			self.running = False
			self.thread.join()

		# relase stream
		if self.vid.isOpened():
			self.vid.release()
 
class tkCamera(tkinter.Frame):

	def __init__(self, tab1, row, col, window, text="", video_source=0, width=None, height=None):
		super().__init__(window)
		
		self.window = window

		self.window.maxsize(1010, 700)
		
		self.window.title(window_title)
		self.video_source = video_source
		self.vid = MyVideoCapture(self.video_source, width, height)

		self.label = tkinter.Label(tab1, text=text)
		self.label.grid(row=row*2, column=col)
		
		self.canvas = tkinter.Canvas(tab1, width=self.vid.width, height=self.vid.height)
		self.canvas.grid(row=row*2 + 1, column=col)
		self.canvas.grid(row=row*2 + 1, column=col)
		 
		# After it is called once, the update method will be automatically called every delay milliseconds
		# calculate delay using `FPS`
		self.delay = int(1000/self.vid.fps)

		self.image = None
		
		self.running = True
		if (self.running):
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

	def __init__(self, window_title):

		self.window = tkinter.Tk()

		self.window.title(window_title)
		
		self.tabControl = ttk.Notebook(self.window)	
		
		self.tab1 = ttk.Frame(self.tabControl, width=1010, height=700)
		self.tab2 = ttk.Frame(self.tabControl, width=1010, height=700)
		self.tab3 = ttk.Frame(self.tabControl, width=1010, height=700)
		self.tabControl.add(self.tab1, text="FACE")
		self.tabControl.add(self.tab2, text="BW")
		self.tabControl.add(self.tab3, text="FLOWER")

		self.vids = []

		sty_tabs = [(self.tab1, "face", "guy"), (self.tab2, "bw", "bo"), (self.tab3, "flower", "boy")]
		for t,s,c in sty_tabs:
			video_sources = get_sources(s, c)
			columns = 3
			for number, source in enumerate(video_sources):
				text, stream = source
				row = number % columns
				col = number // columns
				vid = tkCamera(t, col, row, self.window, text, stream, 300, 300)
				self.vids.append(vid)

		self.tabControl.grid(row=0, column=0)

		self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
		self.window.mainloop()
	
	def on_closing(self, event=None):
		for source in self.vids:
			source.vid.running = False
		print('[App] exit')
		self.window.destroy()

def get_sources(style, cont):
	sources = [
	('Original Video', './input_images/' + cont + '.mp4'),
	('Train', './output_images/' + cont + "_" + style + '_train.avi'),
	('Train Noise', './output_images/' + cont + "_" + style + '_noise.avi'),
	('Train Flow', './output_images/' + cont + "_" + style + '_flow.avi'),
	('Optical Flow', './test_output/' + cont + '_flow.avi'),
	('Confidence Matrix', './test_output/' + cont + '_conf.avi')
	]

	return sources

if __name__ == '__main__':	 
		
	# Create a window and pass it to the Application object
	App("Tkinter and OpenCV")
