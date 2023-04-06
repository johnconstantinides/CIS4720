import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from ImageFunctions import *

class Gui:

    def __init__(self):
        self.root = tk.Tk()
        self.imagepath = None
        self.img = None

        self.root.geometry("1200x1200")
        self.root.title("Image processor")


        #upload button
        button = tk.Button(text="Upload File",command=self.openFile)
        button.pack()


        #scale
        scale_frame_height = tk.Frame(self.root)
        scale_frame_height.pack()
        self.height_change = tk.Text(scale_frame_height,width=9,height=1)
        height_label = tk.Label(scale_frame_height,text="Height")
        height_label.pack(side=tk.LEFT)
        self.height_change.pack(side=tk.RIGHT)

        scale_frame_width = tk.Frame(self.root)
        scale_frame_width.pack()
        self.width_change = tk.Text(scale_frame_width,width=9,height=1)
        width_label = tk.Label(scale_frame_width,text="Width change")
        width_label.pack(side=tk.LEFT)
        self.width_change.pack(side=tk.RIGHT)
        self.scale_button = tk.Button(text="Scale Image",command=self.scaleImage)
        self.scale_button.pack()

        #horizontal 
        flip_frame = tk.Frame(self.root)
        flip_frame.pack()
        self.horizontalFlipButton = tk.Button(flip_frame,text="Flip Horizontally",command=self.horizontalFlip)
        self.horizontalFlipButton.pack(side=tk.LEFT)

        #vertical
        self.verticalFlipButton = tk.Button(flip_frame,text="Flip vertically", command=self.verticalFlip)
        self.verticalFlipButton.pack(side=tk.LEFT)


        #rotation
        rotate_frame = tk.Frame(self.root)
        rotate_frame.pack()
        rotationLabel = tk.Label(rotate_frame,text="Angle")
        rotationLabel.pack(side=tk.LEFT)
        self.rotation_input = tk.Text(rotate_frame,width=9,height=1)
        self.rotation_input.pack(side=tk.RIGHT)
        self.rotationButton = tk.Button(text="Rotate",command=self.rotateImage)
        self.rotationButton.pack()

        #cropping

        #first x,y values
        first_xy = tk.Frame(self.root)
        first_xy.pack()
        x1label = tk.Label(first_xy,text="x1")
        y1label = tk.Label(first_xy,text="y1")
        self.x1 = tk.Text(first_xy,width=5,height=1)
        self.y1 = tk.Text(first_xy,width=5,height=1)
        x1label.pack(side=tk.LEFT)
        self.x1.pack(side=tk.LEFT)
        self.y1.pack(side=tk.RIGHT)
        y1label.pack(side=tk.RIGHT)

        #second x,y values
        second_xy = tk.Frame(self.root)
        second_xy.pack()
        x2label = tk.Label(second_xy,text="x2")
        y2label = tk.Label(second_xy,text="y2")
        self.x2 = tk.Text(second_xy,width=5,height=1)
        self.y2 = tk.Text(second_xy,width=5,height=1)
        x2label.pack(side=tk.LEFT)
        self.x2.pack(side=tk.LEFT)
        self.y2.pack(side=tk.RIGHT)
        y2label.pack(side=tk.RIGHT)
        self.crop_button = tk.Button(text="Crop", command=self.cropImage)
        self.crop_button.pack()

        #convert to grayscale
        grayscale_button = tk.Button(self.root,text="Convert to grayscale", command=self.convertToGrayscale)
        grayscale_button.pack()

        #linear mapping
        linear_frame = tk.Frame(self.root)
        linear_frame.pack()
        gain_label = tk.Label(linear_frame,text="Gain")
        bias_label = tk.Label(linear_frame,text="Bias")
        self.gain_input = tk.Text(linear_frame,width=5,height=1)
        self.bias_input = tk.Text(linear_frame,width=5,height=1)
        gain_label.pack(side=tk.LEFT)
        self.gain_input.pack(side=tk.LEFT)
        bias_label.pack(side=tk.LEFT)
        self.bias_input.pack(side=tk.LEFT)

        linear_button = tk.Button(self.root,text="Apply linear mapping", command=self.linearMapping)
        linear_button.pack()


        #power law mappings
        power_law_frame = tk.Frame(self.root)
        power_law_frame.pack()
        gamma_label = tk.Label(power_law_frame,text="Gamma Value")
        self.gamma_input = tk.Text(power_law_frame,width=5,height=1)
        power_law_button = tk.Button(power_law_frame,text="Apply power law", command=self.powerMapping)
        gamma_label.pack(side=tk.LEFT)
        self.gamma_input.pack(side=tk.LEFT)
        power_law_button.pack(side=tk.LEFT)


        #histogram stuff
        create_histogram = tk.Button(self.root,text="Create histogram",command= lambda: createHistogram(self.img))
        create_histogram.pack()

        #histogram equilization
        equalize_histogram_button = tk.Button(self.root,text="Histogram Equalization",command=self.histogramEqualization)
        equalize_histogram_button.pack()

        #convolution
        convolution_frame = tk.Frame(self.root)
        convolution_frame.pack()
        kernel_label = tk.Label(convolution_frame,text="Enter a MxN odd kernerl")
        self.kernel_input = tk.Text(convolution_frame,width=30,height=1)
        convolution_button = tk.Button(convolution_frame,text="Apply kernel",command=self.convolution)
        kernel_label.pack(side=tk.LEFT)
        self.kernel_input.pack(side=tk.LEFT)
        convolution_button.pack(side=tk.LEFT)


        #non linear filetering
        non_linear_frame = tk.Frame(self.root)
        non_linear_frame.pack()
        filter_label = tk.Label(non_linear_frame,text="Filter size")
        self.filtersize_input = tk.Text(non_linear_frame,width=3,height=1)
        min_filtering_button = tk.Button(non_linear_frame,text="Min filter",command=self.minFiltering)
        max_filtering_button = tk.Button(non_linear_frame,text="Max filter",command=self.maxFiltering)
        med_filtering_button = tk.Button(non_linear_frame,text="Median filter",command=self.medFiltering)
        filter_label.pack(side=tk.LEFT)
        self.filtersize_input.pack(side=tk.LEFT)
        min_filtering_button.pack(side=tk.LEFT)
        max_filtering_button.pack(side=tk.LEFT)
        med_filtering_button.pack(side=tk.LEFT)

        edge_detection_button = tk.Button(self.root,text="Edge detection",command=self.edgeDetection)
        edge_detection_button.pack()



        #image display
        self.image_display = tk.Label()
        self.image_display.pack()

        #main loop
        self.root.mainloop()

    #methods that use the imagefunctions I made and gets them to work with the gui
    #opens file and displaus it
    def openFile(self):
        self.imagepath =  filedialog.askopenfilename()
        self.img = Image.open(self.imagepath)
        self.displayImage()
    
    #displays current image
    def displayImage(self):
        img = ImageTk.PhotoImage(self.img)
        self.image_display.configure(image=img)
        self.image_display.image = img
    
    def scaleImage(self):
        newImg = resizeImage(self.img,float(self.height_change.get(1.0, "end-1c")),float(self.width_change.get(1.0, "end-1c")))
        if newImg:
            self.img = newImg
            self.displayImage()
    
    def horizontalFlip(self):
        newImg = horizontalFlip(self.img)
        if newImg:
            self.img = newImg
            self.displayImage()

    def verticalFlip(self):
        newImg = verticalFlip(self.img)
        if newImg:
            self.img = newImg
            self.displayImage()
    
    def rotateImage(self):
        newImg = rotateImage(self.img,float(self.rotation_input.get(1.0, "end-1c")))
        if newImg:
            self.img = newImg
            self.displayImage()
    
    def cropImage(self):
        newImg = cropImage(self.img,(int(self.x1.get(1.0, "end-1c")),int(self.y1.get(1.0, "end-1c"))),(int(self.x2.get(1.0, "end-1c")),int(self.y2.get(1.0, "end-1c"))))
        if newImg:
            self.img = newImg
            self.displayImage()

    def convertToGrayscale(self):
        newImg = convertToGrayscale(self.img)
        if newImg:
            self.img = newImg
            self.displayImage()
    
    def linearMapping(self):
        newImg = linearMapping(self.img,float(self.gain_input.get(1.0,"end-1c")),float(self.bias_input.get(1.0,"end-1c")))
        if newImg:
            self.img = newImg
            self.displayImage()

    def powerMapping(self):
        newImg = powerMapping(self.img,float(self.gamma_input.get(1.0,"end-1c")))
        if newImg:
            self.img = newImg
            self.displayImage()
        
    def histogramEqualization(self):
        newImg = histogramEqualization(self.img)
        if newImg:
            self.img = newImg
            self.displayImage()
            createHistogram(self.img)

    def convolution(self):
        kernel = eval(self.kernel_input.get(1.0,"end-1c"))
        newImg = convolution(self.img,kernel)
        if newImg:
            self.img = newImg
            self.displayImage()

    def minFiltering(self):
        newImg = minFiltering(self.img,int(self.filtersize_input.get(1.0,"end-1c")))
        if newImg:
            self.img = newImg
            self.displayImage()
    
    def maxFiltering(self):
        newImg = maxFiltering(self.img,int(self.filtersize_input.get(1.0,"end-1c")))
        if newImg:
            self.img = newImg
            self.displayImage()

    def medFiltering(self):
        newImg = medFiltering(self.img,int(self.filtersize_input.get(1.0,"end-1c")))
        if newImg:
            self.img = newImg
            self.displayImage()
    
    def edgeDetection(self):
        newImg = sobelDetection(self.img)
        if newImg:
            self.img = newImg
            self.displayImage()

    
Gui()