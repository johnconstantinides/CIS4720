from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy
import statistics

#resizes an image using nearest-neighbour 
def resizeImage(img,widthChange,heightChange):
    width, height = img.size

    #create image with the new size
    imgNew = Image.new(mode=img.mode,size=(int(width*widthChange),int(height*heightChange)))
    pixelMap = imgNew.load()


    #resizes images using nearest-neighbour
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

    #flips the x coordinates of every pixel
    for i in range(width):
        for j in range(height):
            # flip the i (x) value for the image
            pixelMap[i,j] = img.getpixel((width - 1 - i,j))

    return imgNew

def verticalFlip(img):
    width, height = img.size

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
 
    #flips the y coordinates of everypixel
    for i in range(width):
        for j in range(height):
            # flip the j (y) values for the image
            pixelMap[i,j] = img.getpixel((i,height - 1 -j))

    return imgNew

#crops an image
def cropImage(img,firstPoint, secondPoint):
    width, height = img.size

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
 
    for i in range(width):
        for j in range(height):

            #only add values from image if between the two points
            if i >= firstPoint[0] and i <= secondPoint[0] and j >= firstPoint[1] and j <= secondPoint[1]:
                pixelMap[i,j] = img.getpixel((i,j))

    return imgNew

from PIL import Image

def horizontalShear(img, shear):
    width, height = img.size

    imgNew = Image.new(mode=img.mode, size=(width + int(shear * width), height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            new_i = i + int(j * shear)
            new_j = j

            # Assign the pixel value to the sheared image
            if new_i >= 0 and new_i < width + int(shear * width) and new_j >= 0 and new_j < height:
                pixelMap[new_i, new_j] = img.getpixel((i, j))

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
            #rotate the image around the center of the image
            rotation_x = center[0] + round(math.cos(angle_rad)*(i - center[0]) - math.sin(angle_rad)*(j - center[1])) 
            rotation_y = center[1] + round(math.sin(angle_rad)*(i - center[0]) + math.cos(angle_rad)*(j - center[1]))
            
            if rotation_x >= 0 and rotation_x < width and rotation_y >= 0 and rotation_y < height:
                pixelMap[i,j] = img.getpixel((rotation_x,rotation_y))
            

    return imgNew

#converts rgb images to grayscale
def convertToGrayscale(img):
    if img.mode == "L":
        return img
    
    width, height = img.size
    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            rgb = img.getpixel((i,j))
            
            #convert the rgb value to a single intensity
            pixelMap[i,j] = int((rgb[0] + rgb[1] + rgb[2])/3)
    return imgNew

#applies a linear mapping to grayscale images
def linearMapping(img,a,b):
    if img.mode != "L":
        return
    width, height = img.size
    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            #apply the mapping to each pixel
            pixelMap[i,j] = max(min(int(a*img.getpixel((i,j)) + b),255),0)
    
    return imgNew

#applies a powerMapping to grayscale images
def powerMapping(img, gamma):
    if img.mode != "L":
        return
    
    width, height = img.size
    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            #apply the mapping to each pixel
            pixelMap[i,j] = int(255.0*(pow(img.getpixel((i,j))/255.0,gamma)))
    
    return imgNew


#Ccreates and displays a histogram
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

#creates a normalized histogram of image and returns it
def normalizedHistogram(img):
    if img.mode != 'L':
        return
    
    width, height = img.size

    #create the normalized histogram
    normalized_histogram = numpy.zeros((256), dtype=float)
    for i in range(width):
        for j in range(height):
            normalized_histogram[img.getpixel((i,j))] += 1/(width*height)

    return normalized_histogram

#creates a cumulative normalized histogram from a normalized histogram and returns it
def cumulativeNormalizedHistogram(normalized_histogram):

    #get the sum from 0 to i
    cumulative_normalized_histogram = numpy.zeros((256), dtype=float)
    for i in range(0,len(normalized_histogram)):
        cumulative_normalized_histogram[i] = sum(normalized_histogram[0:i + 1]) 

    return cumulative_normalized_histogram



def histogramEqualization(img):
    if img.mode != 'L':
        return
    #counts the ammount of pixels for each intensity level
    width, height = img.size
    normalized_histogram = normalizedHistogram(img)

    #get the cumulative normalized histogram
    cumulative_normalized_histogram = cumulativeNormalizedHistogram(normalized_histogram)


    imgNew = Image.new(mode="L",size=(width,height))
    pixelMap = imgNew.load()

    #use the cumulative normalized histogram to adjust intensity levels
    for i in range(width):
        for j in range(height):
            pixelMap[i,j] = int(cumulative_normalized_histogram[img.getpixel((i,j))]*255)
    
    return imgNew

def convolution(img,kernel):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    #get lenght and width of kernel. if either are even return since they should be odds
    m = len(kernel)
    n = len(kernel[0])
    if(m % 2 == 0 or m % 2 == 0):
        print('Kernel should have odd dimensions')
        return

    #loop through each pixel in the image
    for x in range(width):
        for y in range(height):

            #This allows the function to work with both grayscale and rgb images
            if img.mode == 'L':
                pixelMap[x,y] = 0
            elif img.mode == 'RGB':
                pixelMap[x,y] = (0,0,0)

            #convolution of each image
            for i in range(-(m - 1)//2, (m - 1)//2 + 1):
                for j in range(-(n - 1)//2, (n - 1)//2 + 1):

                    #if current pixel is not in image then dont add as it would just be zero zince assuming zero padding
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        
                        #adjust i and j since so they can match the indices of the kernel.
                        if img.mode == 'L':
                            #add the value to the pixel
                            pixelMap[x,y] += int(img.getpixel((x - i,y - j))*kernel[i + (m - 1)//2][j + (n - 1)//2])
                        elif img.mode == 'RGB':
                            #get the kernel value
                            kernel_value = kernel[i + (m - 1)//2][j + (n - 1)//2]
                            #apply the kernel value to each rgb value in the pixel
                            newPixel = tuple([pixel_value * kernel_value for pixel_value in img.getpixel((x - i, y - j))])
                            #add the value to the current rgb pixel
                            pixelMap[x,y] = tuple(int(pp) for pp in map(sum,zip(pixelMap[x,y],newPixel)))

    return imgNew


def minFiltering(img,filterSize):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for x in range(width):
        for y in range(height):

            pixelMap[x,y] = img.getpixel((x,y))

            if img.mode == 'L':
                minimum = 255
            else:
                minimum = (255,255,255)

            #loop through the kernel
            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #make sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        #compare the values to the minimum. If smaller the replace them
                        if img.mode == 'L':
                            minimum = min(minimum,img.getpixel((x - i, y - j)))
                        else:
                            r = min(minimum[0],img.getpixel((x - i, y - j))[0])
                            g = min(minimum[1],img.getpixel((x - i, y - j))[1])
                            b = min(minimum[2],img.getpixel((x - i, y - j))[2])
                            minimum = (r,g,b)
            #set pixel with the minimum value
            pixelMap[x,y] = minimum


    return imgNew

def maxFiltering(img,filterSize):
    if img.mode not in ['L','RGB']:
        print('Only works with Grayscale or rgb images')
        return
    
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for x in range(width):
        for y in range(height):

            pixelMap[x,y] = img.getpixel((x,y))
            if img.mode == 'L':
                maximum = 0
            else:
                maximum = (0,0,0)
            
            #loop through the kernel
            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #make sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):

                        #compare the values to the maximum. If greater the replace them
                        if img.mode == 'L':
                            maximum = min(maximum,img.getpixel((x - i, y - j)))
                        else:
                            r = max(maximum[0],img.getpixel((x - i, y - j))[0])
                            g = max(maximum[1],img.getpixel((x - i, y - j))[1])
                            b = max(maximum[2],img.getpixel((x - i, y - j))[2])
                            maximum = (r,g,b)
            #set pixel with the maximum value
            pixelMap[x,y] = maximum


    return imgNew

def medFiltering(img,filterSize):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for x in range(width):
        for y in range(height):

            pixelMap[x,y] = img.getpixel((x,y))

            if img.mode == 'L':
                values = []
            elif img.mode == 'RGB':
                values = [[],[],[]]
            
            #loop through the neighbourhood of pixel (x,y)
            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #make sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):

                        #add each value to the array values
                        if img.mode == 'L':
                            values.append(img.getpixel((x - i, y - j)))
                        else:
                            values[0].append(img.getpixel((x - i, y - j))[0])
                            values[1].append(img.getpixel((x - i, y - j))[1])
                            values[2].append(img.getpixel((x - i, y - j))[2])
            
            #compute the median value and update the current pixel to it
            if img.mode == 'L':
                pixelMap[x,y] = int(statistics.median(values))
            elif img.mode == 'RGB':
                pixelMap[x,y] = (int(statistics.median(values[0])),int(statistics.median(values[1])),int(statistics.median(values[2])))

    return imgNew
    
def sobelDetection(img):
    width, height = img.size

    #convert image to a grayscale image if it is not already one
    if img.mode != 'L':
        img = convertToGrayscale(img)
    imgNew = Image.new(mode='L',size=(width,height))
    pixelMap = imgNew.load()

    #vertical lines
    x_image = convolution(img,[[1,0,-1],[2,0,-2],[1,0,-1]])
    #horizontal
    y_image = convolution(img,[[1,2,1],[0,0,0],[-1,-2,-1]])

    #get the gradient magnitude
    for i in range(width):
        for j in range(height):
            pixelMap[i,j] =  int(math.sqrt(x_image.getpixel((i,j))**2 + y_image.getpixel((i,j))**2))
    
    #thresholding.
    for i in range(width):
        for j in range(height):
            if imgNew.getpixel((i,j)) > 30:
                pixelMap[i,j] = 255
            else:
                pixelMap[i,j] = 0

    return imgNew
