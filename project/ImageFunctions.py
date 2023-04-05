from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy
import statistics

def resizeImage(img,widthChange,heightChange):
    width, height = img.size

    #create image with the new size
    imgNew = Image.new(mode=img.mode,size=(int(width*widthChange),int(height*heightChange)))
    pixelMap = imgNew.load()


    for i in range(int(width*widthChange)):
        for j in range(int(height*heightChange)):
            newX = i / widthChange
            newY = j / heightChange
            # print(newX,newY)
            pixelMap[i,j] = img.getpixel((int(newX),int(newY)))
    
    return imgNew


def horizontalFlip(img):
    width, height = img.size

    print(img.mode)

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            # print(newX,newY)
            pixelMap[i,j] = img.getpixel((width - 1 - i,j))

    return imgNew

def verticalFlip(img):
    width, height = img.size

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
 

    for i in range(width):
        for j in range(height):
            # print(newX,newY)
            pixelMap[i,j] = img.getpixel((i,height - 1 -j))

    return imgNew

def cropImage(img,firstPoint, secondPoint):
    width, height = img.size

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
 

    for i in range(width):
        for j in range(height):
            if i >= firstPoint[0] and i <= secondPoint[0] and j >= firstPoint[1] and j <= secondPoint[1]:
                pixelMap[i,j] = img.getpixel((i,j))

    return imgNew


def rotateImage(img,angle):
    width, height = img.size

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
    
    # find the center of the image
    center = (width//2,height//2)
    angle_rad = math.radians(angle)

    for i in range(width):
        for j in range(height):
            #rotation but adjust for it to be around the center of the image
            rotation_x = center[0] + int(math.cos(angle_rad)*(i - center[0]) - math.sin(angle_rad)*(j - center[1])) 
            rotation_y = center[1] + int(math.sin(angle_rad)*(i - center[0]) + math.cos(angle_rad)*(j - center[1]))
            if rotation_x >= 0 and rotation_x < width and rotation_y >= 0 and rotation_y < height:
                pixelMap[i,j] = img.getpixel((rotation_x,rotation_y))
            

    return imgNew

def convertToGrayscale(img):
    if img.mode == "L":
        return
    
    width, height = img.size
    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            rgb = img.getpixel((i,j))
            pixelMap[i,j] = int((rgb[0] + rgb[1] + rgb[2])/3)
    return imgNew

def linearMapping(img,a,b):
    if img.mode != "L":
        return
    width, height = img.size
    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            pixelMap[i,j] = max(min(int(a*img.getpixel((i,j)) + b),255),0)
    
    return imgNew

def powerMapping(img, gamma):
    if img.mode != "L":
        return
    
    width, height = img.size
    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            pixelMap[i,j] = int(255.0*(pow(img.getpixel((i,j))/255.0,gamma)))
    
    return imgNew


def createHistogram(img):
    if img.mode != 'L':
        return
    
    #counts the ammount of pixels for each intensity level
    width, height = img.size
    intensity_freq = numpy.zeros((256), dtype=int)
    for i in range(width):
        for j in range(height):
            intensity_freq[img.getpixel((i,j))] += 1
    
    #displays the histogram
    intensity_values = numpy.arange(0,256,1)
    plt.bar(intensity_values,intensity_freq,width=.5)
    plt.xlabel("Intensity Levels")
    plt.ylabel("Frequency")
    plt.show()

def histogramEqualization(img):
    if img.mode != 'L':
        return
    #counts the ammount of pixels for each intensity level
    width, height = img.size
    normalized_histogram = numpy.zeros((256), dtype=float)
    for i in range(width):
        for j in range(height):
            normalized_histogram[img.getpixel((i,j))] += 1/(width*height)


    #get the cumulative normalized histogram
    cumulative_normalized_histogram = numpy.zeros((256), dtype=float)
    for i in range(0,len(normalized_histogram)):
        cumulative_normalized_histogram[i] = sum(normalized_histogram[0:i + 1]) 


    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    #use the cumulative normalized histogram to adjust intensity levels
    for i in range(width):
        for j in range(height):
            pixelMap[i,j] = int(cumulative_normalized_histogram[img.getpixel((i,j))]*255)
def convolution(img,kernel):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
    m = len(kernel)
    n = len(kernel[0])
    if(m % 2 == 0 or m % 2 == 0):
        return

    #loop through each pixel in the image
    for x in range(width):
        for y in range(height):
            if img.mode == 'L':
                pixelMap[x,y] = 0
            elif img.mode == 'RGB':
                pixelMap[x,y] = (0,0,0)

            #convulution of each image
            for i in range(-(m - 1)//2, (m - 1)//2 + 1):
                for j in range(-(n - 1)//2, (n - 1)//2 + 1):

                    #if current pixel is not in image then dont add as it would just be zero zince assuming zero padding
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        
                        #adjust i and j since so they can match the indices of the kernel.
                        if img.mode == 'L':
                            pixelMap[x,y] += int(img.getpixel((x - i,y - j))*kernel[i + (m - 1)//2][j + (n - 1)//2])
                        elif img.mode == 'RGB':
                            kernel_value = kernel[i + (m - 1)//2][j + (n - 1)//2]
                            newPixel = tuple([pixel_value * kernel_value for pixel_value in img.getpixel((x - i, y - j))])
                            pixelMap[x,y] = tuple(int(pp) for pp in map(sum,zip(pixelMap[x,y],newPixel)))



    return imgNew


def minFiltering(img,filterSize):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for x in range(width):
        for y in range(height):

            pixelMap[x,y] = img.getpixel((x,y))
            minimum = 255
            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #cmake sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        minimum = min(minimum,img.getpixel((x - i, y - j)))
            pixelMap[x,y] = minimum


    return imgNew

def maxFiltering(img,filterSize):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for x in range(width):
        for y in range(height):

            pixelMap[x,y] = img.getpixel((x,y))
            maximum = 0
            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #cmake sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        maximum = max(maximum,img.getpixel((x - i, y - j)))
            pixelMap[x,y] = maximum


    return imgNew

def medFiltering(img,filterSize):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for x in range(width):
        for y in range(height):

            pixelMap[x,y] = img.getpixel((x,y))
            values = []
            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #cmake sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        values.append(img.getpixel((x - i, y - j)))
            pixelMap[x,y] = int(median(values))


    return imgNew
    