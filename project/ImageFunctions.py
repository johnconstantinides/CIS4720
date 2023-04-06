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
            # flip the i (x) value for the image
            pixelMap[i,j] = img.getpixel((width - 1 - i,j))

    return imgNew

def verticalFlip(img):
    width, height = img.size

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
 

    for i in range(width):
        for j in range(height):
            # flip the j (y) values for the image
            pixelMap[i,j] = img.getpixel((i,height - 1 -j))

    return imgNew

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


def rotateImage(img,angle):
    orginal_width, orginal_height = img.size

    
    # find the center of the image
    center = (orginal_width//2,orginal_height//2)
    angle_rad = math.radians(angle)

    #calculate new width, height after rotation
    p1 = (center[0] + int(math.cos(angle_rad)*(0 - center[0]) - math.sin(angle_rad)*(0 - center[1])),center[1] + int(math.sin(angle_rad)*(0 - center[0]) + math.cos(angle_rad)*(0 - center[1])))
    p2 = (center[0] + int(math.cos(angle_rad)*(orginal_width - 1 - center[0]) - math.sin(angle_rad)*(0 - center[1])),center[1] + int(math.sin(angle_rad)*(orginal_width - 1 - center[0]) + math.cos(angle_rad)*(0 - center[1])))
    p3 = (center[0] + int(math.cos(angle_rad)*(0 - center[0]) - math.sin(angle_rad)*(orginal_height - 1 - center[1])),center[1] + int(math.sin(angle_rad)*(0 - center[0]) + math.cos(angle_rad)*(orginal_height - 1 - center[1])))
    p4 = (center[0] + int(math.cos(angle_rad)*(orginal_width - 1 - center[0]) - math.sin(angle_rad)*(orginal_height - 1 - center[1])),center[1] + int(math.sin(angle_rad)*(orginal_width - 1 - center[0]) + math.cos(angle_rad)*(orginal_height - 1 - center[1])))
    width = max(p1[0],p2[0],p3[0],p4[0]) - min(p1[0],p2[0],p3[0],p4[0])
    height = max(p1[1],p2[1],p3[1],p4[1]) - min(p1[1],p2[1],p3[1],p4[1])
    center = (width//2,height//2)
    

    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    for i in range(width):
        for j in range(height):
            #rotation the image around the center of the image
            rotation_x = center[0] + int(math.cos(angle_rad)*(i - center[0]) - math.sin(angle_rad)*(j - center[1])) 
            rotation_y = center[1] + int(math.sin(angle_rad)*(i - center[0]) + math.cos(angle_rad)*(j - center[1]))
            
            if rotation_x >= 0 and rotation_x < width and rotation_y >= 0 and rotation_y < height and rotation_x < orginal_width and rotation_y < orginal_height:
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
            
            #convert the rgb value to a single intensity
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
            if img.mode == 'L':
                minimum = 255
            else:
                minimum = (255,255,255)

            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #cmake sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        if img.mode == 'L':
                            minimum = min(minimum,img.getpixel((x - i, y - j)))
                        else:
                            r = min(minimum[0],img.getpixel((x - i, y - j))[0])
                            g = min(minimum[1],img.getpixel((x - i, y - j))[1])
                            b = min(minimum[2],img.getpixel((x - i, y - j))[2])
                            minimum = (r,g,b)
            pixelMap[x,y] = minimum


    return imgNew

def maxFiltering(img,filterSize):
    if img.mode not in ['L','RGB']:
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
            for i in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                for j in range(-(filterSize - 1)//2,(filterSize - 1)//2 + 1):
                    #cmake sure pixel is not out of the ranges
                    if not ((x - i) < 0 or (x - i) >= width or (y - j) < 0 or (y - j) >= height):
                        if img.mode == 'L':
                            maximum = min(maximum,img.getpixel((x - i, y - j)))
                        else:
                            r = max(maximum[0],img.getpixel((x - i, y - j))[0])
                            g = max(maximum[1],img.getpixel((x - i, y - j))[1])
                            b = max(maximum[2],img.getpixel((x - i, y - j))[2])
                            maximum = (r,g,b)
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
    
def sobelDetection(img):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()

    #vertical lines
    x_image = convolution(img,[[1,0,-1],[2,0,-2],[1,0,-1]])
    y_image = convolution(img,[[-1,-2,-1],[0,0,0],[1,2,1]])
    average = 0
    for i in range(width):
        for j in range(height):
            if img.mode == 'L':
                average += int(math.sqrt(x_image.getpixel((i,j))**2 + y_image.getpixel((i,j))**2))


    average = int(average/(width*height))

    for i in range(width):
        for j in range(height):
            if img.mode == 'L':
                magnitude = int(math.sqrt(x_image.getpixel((i,j))**2 + y_image.getpixel((i,j))**2))
                if magnitude > average:
                    pixelMap[i,j] = 255
                else:
                    pixelMap[i,j] = 0

    return imgNew

def laplacianEdgeDetection(img):
    width, height = img.size
    imgNew = Image.new(mode=img.mode,size=(width,height))
    pixelMap = imgNew.load()
    
    pp = convolution(img,[[0,1,0],[1,-4,1],[0,1,0]])
    average = 0
    for i in range(width):
        for j in range(height):
            if img.mode == 'L':
                average += pp.getpixel((i,j))

    average = int(average/(width*height))

    for i in range(width):
        for j in range(height):
            if img.mode == 'L':
                if pp.getpixel((i,j)) > average:
                    pixelMap[i,j] = 255
                else:
                    pixelMap[i,j] = 0

    return imgNew