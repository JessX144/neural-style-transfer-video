from queue import Queue, Empty

import threading

# import tkinter and all its functions
from tkinter import * 
from PIL import ImageTk, Image
import tkinter as tk
import cv2
import subprocess
from tkinter import ttk
from pathlib import Path

def myfunc(self, sty_type):
	if (len(self.textBox.get("1.0", END)) == 1):
		# print('python stylise.py -s ' + self.var_style.get() + ' -m ' + sty_type + ' -c ' + self.var_video.get())
		subprocess.call('python stylise.py -s ' + self.var_style.get() + ' -m ' + sty_type + ' -c ' + self.var_video.get())
	else:
		# print('python stylise.py -s ' + self.var_style.get() + ' -m ' + sty_type + ' -u ' + self.textBox.get("1.0",END))
		subprocess.call('python stylise.py -s ' + self.var_style.get() + ' -m' + sty_type + ' -u ' + self.textBox.get("1.0",END))
	self.queue.put("Task finished")

class MainWindow():
	def __init__(self, window, cap):
		self.vid_running = True
		self.basewidth = 350

		self.window = window
		self.window.title("Neural Style Transfer") # title of the GUI window
		self.window.geometry('1155x550')
		self.window.maxsize(1200, 550) # specify the max size the window can expand to
		self.window.config(bg="skyblue") # specify background color

		# Create left and right frames
		self.left_frame = Frame(self.window, width=400, height=400, bg='grey')
		self.left_frame.grid(row=0, column=0, padx=10, pady=5)

		self.right_frame = Frame(self.window, width=400, height=400, bg='grey')
		self.right_frame.grid(row=0, column=1, padx=10, pady=5)
		self.bottom_frame = Frame(self.window, width=900, height=100, bg='grey')
		self.bottom_frame.grid(row=1, column=0, padx=10, pady=5, columnspan = 3, sticky = tk.W+tk.E)

		self.out_frame = Frame(self.window, width=400, height=400, bg='grey')
		self.out_frame.grid(row=0, column=2, padx=10, pady=5)
		self.out_vid_lab = Label(self.out_frame, text="Styled Video")
		self.out_vid_lab.grid(row=0, column=2, padx=5, pady=5)
		img = Image.open("./input_images/grey.jpg")
		img = res_im(self.basewidth, img)
		image = ImageTk.PhotoImage(img)
		self.out_vid = Label(self.out_frame, text="Styled Image", image=image)
		self.out_vid.grid(row=1, column=2, padx=5, pady=5)		 

		self.cap = cap
		self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.interval = 20 # Interval in ms to get the latest frame

		self.sty_im_lab = Label(self.left_frame, text="Style Image")
		self.sty_im_lab.grid(row=0, column=0, padx=5, pady=5)
		self.sty_vid_lab = Label(self.right_frame, text="Style Video").grid(row=0, column=0, padx=5, pady=5)

		self.var_style = StringVar(None, "bw")
		self.var_video = StringVar(None, "guy")

		img = Image.open("./style_images/" + self.var_style.get() + ".jpg")
		img = res_im(self.basewidth, img)
		image = ImageTk.PhotoImage(img)  
		
		self.sty_im = Label(self.left_frame, text="Style Image", image=image)
		self.sty_im.grid(row=1, column=0, padx=5, pady=5)		

		self.sty_vid = Label(self.right_frame, image=image)
		self.sty_vid.grid(row=1, column=0, padx=5, pady=5)

		self.frame_counter = 0
		self.frame_counter_out = 0

		# Update image on canvas
		self.update_image()

		self.label = Label(self.right_frame, text="Youtube URL:")
		self.label.grid(row=3, column=0, padx=5, pady=5)

		self.textBox=Text(self.right_frame, height=1, width=30)
		self.textBox.grid(row=4, column=0, padx=5, pady=5)

		self.sty = Button(self.bottom_frame, text="Default Stylise", command=lambda: self.stylise("train"))
		self.sty.grid(row=0, column=0, padx=5, pady=5)	
		self.sty_flow = Button(self.bottom_frame, text="Optical Flow Stylise", command=lambda: self.stylise("flow"))
		self.sty_flow.grid(row=0, column=1, padx=5, pady=5)	
		self.sty_noise = Button(self.bottom_frame, text="Noisy Stylise", command=lambda: self.stylise("noise"))
		self.sty_noise.grid(row=0, column=2, padx=5, pady=5)	

		# Create tool bar frame
		self.styles_bar = Frame(self.left_frame, width=180, height=185)
		self.styles_bar.grid(row=2, column=0, padx=5, pady=5)

		R1 = Radiobutton(self.styles_bar, text="BW", variable=self.var_style, value='bw', command=self.sel_style).grid(row=0, column=0, padx=5, pady=3, ipadx=10) 
		R2 = Radiobutton(self.styles_bar, text="face", variable=self.var_style, value='face', command=self.sel_style).grid(row=0, column=1, padx=5, pady=3, ipadx=10) 
		R3 = Radiobutton(self.styles_bar, text="keefe", variable=self.var_style, value='keefe', command=self.sel_style).grid(row=0, column=2, padx=5, pady=3, ipadx=10) 

		video_bar = Frame(self.right_frame, width=180, height=185)
		video_bar.grid(row=2, column=0, padx=5, pady=5)

		R4 = Radiobutton(video_bar, text="Bo", variable=self.var_video, value='bo', command=self.sel_vid).grid(row=0, column=0, padx=5, pady=3, ipadx=10) 
		R5 = Radiobutton(video_bar, text="Boy", variable=self.var_video, value='boy', command=self.sel_vid).grid(row=0, column=1, padx=5, pady=3, ipadx=10) 
		R6 = Radiobutton(video_bar, text="Guy", variable=self.var_video, value='guy', command=self.sel_vid).grid(row=0, column=2, padx=5, pady=3, ipadx=10) 

		self.sel_style()

	def stylise(self, sty_type):
		self.sty_type = sty_type
		self.progress()
		self.prog_bar.start()
		self.queue = Queue()
		thread = threading.Thread(target=myfunc, args=[self, sty_type])
		thread.start()
		self.window.after(100, self.process_queue)

	def progress(self):
		self.prog_bar = ttk.Progressbar(
			self.bottom_frame, orient="horizontal",
			length=200, mode="indeterminate"
			)
		self.prog_bar.grid(row=0, column=3, padx=5, pady=5)	

	def update_output(self):
		if (self.vid_running):
			ret, frame = self.cap_out.read()
			self.frame_counter_out += 1

			if self.frame_counter_out == self.cap_out.get(cv2.CAP_PROP_FRAME_COUNT):
				self.frame_counter_out = 0
				self.cap_out.set(cv2.CAP_PROP_POS_FRAMES, 0)

			# Get the latest frame and convert image format
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to RGB
			image = Image.fromarray(image) # to PIL format
			image = res_im(self.basewidth, image)
			image = ImageTk.PhotoImage(image) # to ImageTk format
		
			# Update image
			self.out_vid.image = image
			self.out_vid.configure(image=image)
			self.out_vid.grid(row=1, column=2, padx=5, pady=5)		

			# Repeat every 'interval' ms
			if (self.vid_running):
				self.window.after(self.interval, self.update_output)

	def process_queue(self):
		try:
			msg = self.queue.get(0)
			if (len(self.textBox.get("1.0", END)) != 1):
				if (Path('./input_images/video.mkv').is_file()):
					self.cap = cv2.VideoCapture('./input_images/video.mkv')
				else:
					self.cap = cv2.VideoCapture('./input_images/video.mp4')
				self.cap_out = cv2.VideoCapture('./output_images/video_' + self.var_style.get() + '_' + self.sty_type + '.avi')
			else:
				self.cap_out = cv2.VideoCapture('./output_images/' + self.var_video.get() + '_' + self.var_style.get() + '_' + self.sty_type + '.avi')
			self.prog_bar.stop()
			self.prog_bar.destroy()
			self.update_output()
		except Empty:
			if (self.vid_running):
				self.window.after(100, self.process_queue)

	def update_image(self):
		if (self.vid_running):
			ret, frame = self.cap.read()
			self.frame_counter += 1

			if self.frame_counter == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
				self.frame_counter = 0
				self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

			# Get the latest frame and convert image format
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to RGB
			image = Image.fromarray(image) # to PIL format
			image = res_im(self.basewidth, image)
			image = ImageTk.PhotoImage(image) # to ImageTk format
		
			# Update image
			self.sty_vid.image = image
			self.sty_vid.configure(image=image)
			self.sty_vid.grid(row=1, column=0, padx=5, pady=5)

			# Repeat every 'interval' ms
			if (self.vid_running):
				self.window.after(self.interval, self.update_image)

	def quit(self):
		self.root.destroy()

	def sel_style(self):
		# self.label.config(text = "Style {} video with style {}".format(self.var_video.get(), self.var_style.get()))
		img = Image.open("./style_images/" + self.var_style.get() + ".jpg")

		img = res_im(self.basewidth, img)
		imag = ImageTk.PhotoImage(img)

		self.sty_im.image = imag
		self.sty_im.configure(image=imag)
		self.sty_im.grid(row=1, column=0, padx=5, pady=5)

	def sel_vid(self):
		# self.label.config(text = "Style {} video with style {}".format(self.var_video.get(), self.var_style.get()))
		self.cap = cv2.VideoCapture('./input_images/' + self.var_video.get() + '.mp4')
		self.frame_counter = 0

def res_im(basewidth, img):
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	return img 

root = tk.Tk()

MainWindow(root, cv2.VideoCapture('./input_images/guy.mp4'))

root.mainloop()