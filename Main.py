import numpy as np
import cv2
import time
import os
import math
import subprocess
from enum import Enum
IMAGEDIR = './input_images/'

import subprocess



def getImageFromListOfImages(listofimages, want):
    for imagepath in listofimages:
        print(imagepath)
        if(want in imagepath):
            return imagepath

    raise Exception('Could not find image with this name!')


def getAllImagesFromInputImagesDir(path:str, getabspaths=True):
    listOfImagePaths = []

    if (path[0] == '.' and getabspaths):
        path = os.getcwd() + path[1:path.__len__()]


    # read the entries
    with os.scandir(path) as listOfEntries:
        curr_path = ""
        for entry in listOfEntries:
            # print all entries that are files
            if entry.is_file() and ('png' in entry.name.lower() or 'jpg' in entry.name.lower()):
                #print(entry.name)

                if (getabspaths):
                    curr_path=path+entry.name
                    #print(path)
                else:
                    curr_path = entry.name


                listOfImagePaths.append(curr_path)

    return listOfImagePaths






def displayImageGivenPath(imagepath:str, wantGrayImage=False):
    img = getImageArray(imagepath,wantGrayImage)
    # print(img)
    # print(img[0])
    # print(len(img)) #prints number of rows (image pixel height)
    # print(len(img[0])) #prints number of columns (image pixel width)
    #
    # print(img[0])
    # print(img[0][0])
    cv2.imshow('image', img)
    cv2.waitKey(0) #waits for user to type in a key
    cv2.destroyAllWindows()

def displayImageGivenArray(numpyimagearr):
    cv2.imshow('image', numpyimagearr)
    cv2.waitKey(0)  # waits for user to type in a key
    cv2.destroyAllWindows()

def getImageArray(imagepath:str, intensitiesOnly=True):
    if(intensitiesOnly):
        return cv2.imread(imagepath, 0)
    else:
        return cv2.imread(imagepath)

#Part 1:------------------------------
'''
I is image of varying size, 
H is kernel of varying size. 
Output: convolution result that is displayed. 
'''
def Convolve(I,H):
    newimage = np.copy(I)
    isColorImage = False

    imagedimensions = newimage.shape
    image_height = imagedimensions[0]
    image_width = imagedimensions[1]


    if(isinstance(I[0][0],np.uint8)):
        isColorImage = False
        print("THIS IS A GRAYSCALE IMAGE")
    else:
        isColorImage = True
        print("THIS IS A COLOR IMAGE")

    print(newimage)

    print(image_width,image_height)

    centerOfKernel = -1

    kerneldimensions = H.shape
    kernel_height = kerneldimensions[0]
    kernel_width = kerneldimensions[1]

    middleRow = math.floor(kernel_height/2)
    middleColumn = math.floor(kernel_width/2)
    kernelTotal = np.sum(a=H)
    print("KERNEL TOTAL:"+str(kernelTotal))
    #print(middleRow,middleColumn)

    if(middleRow==middleColumn):
        centerOfKernel = middleColumn
    else:
        raise Exception('The kernel is not a perfect square! ERROR')
        return None

    num_pixels = 0
    for row_index in range(0,image_height):

        for column_index in range(0,image_width):
            # if(row_index>=280 and column_index>=380):
            #     print("ENTERED")
            #     ''

            num_pixels+=1



            #Calculates the new intensity value:
            summedKernelIntensityValues = 0

            summedKernelBlueValues = 0
            summedKernelGreenValues = 0
            summedKernelRedValues= 0

            for kernel_row_index in range(0,kernel_height):


                for kernel_column_index in range(0,kernel_width):

                    #currentKernelValue=H[kernel_row_index][kernel_column_index]
                    currentKernelValue = H.item((kernel_row_index,kernel_column_index))


                    #CV2 Uses BGR array layout for colors
                    currentBluePixelValue = 0
                    currentGreenPixelValue = 0
                    currentRedPixelValue = 0


                    currentRowKernelLinedAgainstImage = row_index+(kernel_row_index-middleRow)
                    currentColumnKernelLinedAgainstImage = column_index+(kernel_column_index-middleColumn)
                    #print("CURRENT ROW, COLUMN INDEX:"+str(row_index)+","+str(column_index))
                    #print("CURRENT KERNEL LINED AGAINST IMAGE INDEX:"+str(currentRowKernelLinedAgainstImage)+","+str(currentColumnKernelLinedAgainstImage))


                    if(currentRowKernelLinedAgainstImage>=0 and currentRowKernelLinedAgainstImage<image_height and currentColumnKernelLinedAgainstImage>=0 and currentColumnKernelLinedAgainstImage<image_width):
                        #print("LINING UP KERNEL AGAINST:"+str(currentRowKernelLinedAgainstImage)+","+str(currentColumnKernelLinedAgainstImage))
                        #currentIntensityPixelValue = I[currentRowKernelLinedAgainstImage][currentColumnKernelLinedAgainstImage]

                        if(isColorImage==False):
                            #print("ENTERED")
                            currentIntensityPixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage)
                            newIntensityValue = int((currentIntensityPixelValue * currentKernelValue))
                            summedKernelIntensityValues += newIntensityValue

                        elif (isColorImage):
                            currentBluePixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage,0)
                            newBluePixelValue = int((currentBluePixelValue * currentKernelValue))
                            summedKernelBlueValues+=newBluePixelValue

                            currentGreenPixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage,1)
                            newGreenPixelValue = int((currentGreenPixelValue * currentKernelValue))
                            summedKernelGreenValues += newGreenPixelValue

                            currentRedPixelValue = I.item(currentRowKernelLinedAgainstImage,currentColumnKernelLinedAgainstImage,2)
                            newRedPixelValue = int((currentRedPixelValue * currentKernelValue))
                            summedKernelRedValues += newRedPixelValue



                    else:
                        ''



                        #print("COULD NOT LINE KERNEL AGAINST INVALID INDEX:"+str(currentRowKernelLinedAgainstImage)+","+str(currentColumnKernelLinedAgainstImage))


                    #print("KROW, KCOL:" + str(kernel_row_index) + "," + str(kernel_column_index) + "|CURRENT INTENSITY:" + str(currentIntensityPixelValue) + "|CURRENT KERNEL VALUE:" + str(currentKernelValue) + "|NEW INTENSITY VALUE:" + str(newIntensityValue))

                    #print("CURRENT SUMMED VALUE:"+str(summedKernelIntensityValues))

            #Making sure the kernel total is atleast 1 (to not divide by 0 when using sobel edge kernels)
            if (kernelTotal <= 0):
                kernelTotal = 1
                print("ENTERED KERNEL TOTAL")

            if(isColorImage==False):
                summedKernelIntensityValues = int(summedKernelIntensityValues / kernelTotal)

                # Making sure summedKernelIntensityValues variable is within [0,255]
                if (summedKernelIntensityValues >= 255):
                    summedKernelIntensityValues = 255
                if (summedKernelIntensityValues <= 0):
                    summedKernelIntensityValues = 0

                newimage.itemset((row_index, column_index), int(summedKernelIntensityValues))

            elif(isColorImage):
                summedKernelBlueValues = int(summedKernelBlueValues/kernelTotal)
                summedKernelGreenValues = int(summedKernelGreenValues/kernelTotal)
                summedKernelRedValues = int(summedKernelRedValues/kernelTotal)

                # Making sure summedColorValues are within [0,255]
                if (summedKernelBlueValues >= 255):
                    summedKernelBlueValues = 255
                if (summedKernelBlueValues <= 0):
                    summedKernelBlueValues = 0

                if (summedKernelGreenValues >= 255):
                    summedKernelGreenValues = 255
                if (summedKernelGreenValues <= 0):
                    summedKernelGreenValues = 0

                if (summedKernelRedValues >= 255):
                    summedKernelRedValues = 255
                if (summedKernelRedValues <= 0):
                    summedKernelRedValues = 0

                newimage.itemset((row_index, column_index,0), summedKernelBlueValues)
                newimage.itemset((row_index, column_index, 1), summedKernelGreenValues)
                newimage.itemset((row_index, column_index, 2), summedKernelRedValues)



            #print("---------------------------------------------------")


    # print("HEIGHT:"+str(len(I)))
    # print("WIDTH:"+str(len(I[0])))
    # print("ACTUAL:"+str(len(I)*len(I[0])))
    # print("TOTAL PIXELS:"+str(num_pixels))

    # if(num_pixels==len(I)*len(I[0])):
    #     print("EQUAL")

    #print(newimage)

    return newimage




'''
takes image I as input and outputs copy of image re-sampled
by half the width and height of the input. Remember to 
Gaussian filter the image before reducing it; use separable
1D Gaussian kernels. 
'''
def Reduce(I,factor=0.5):
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])


    new_image = Convolve(I,GAUSSIAN_KERNEL) #Blurs image using Gaussian Blur kernel
    new_image = Expand(new_image,factor)#Scales down image

    return new_image

'''
takes image I as input and outputs copy of image expanded,
twice the width and height of the input. '''
def Expand(I, scaleFactor):
    new_width = int(I.shape[1]*scaleFactor)
    new_height = int(I.shape[0]*scaleFactor)
    #print("EXPANDING TO WIDTH:"+str(new_width)+", HEIGHT:"+str(new_height))
    new_image = cv2.resize(I,(new_width,new_height),interpolation=cv2.INTER_NEAREST)

    #print(new_image)

    return new_image


'''
takes in an image I,
takes in N (int) is the no. of levels. 
'''
def GaussianPyrmaid(I,N):
    curr_image = I
    for i in range(N):
        curr_image = Reduce(curr_image)
        print(curr_image)
        displayImageGivenArray(curr_image)

'''
function which collapses the Laplacian pyramid LI of nlevels 
to generate the original image. Report the error in reconstruction
using image difference. 
'''
def Reconstruct(Ll,N):
    return


def main():

    listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR,True)
    #print(getAllImagesFromInputImagesDir(IMAGEDIR,True))

    #displayImageGivenPath(listOfImages[0])
    # displayImageGivenPath(listOfImages[1])
    # displayImageGivenPath(listOfImages[2])

    # x = np.array([2,3,1,0])
    #
    # print(x)
    # x[2] = 5435
    # print(len(x))
    #
    # t1 = time.time()
    # yb = range(1000)
    # for i in yb:
    #     ''
    #     #print(x)
    # print("PYTHON:"+str(time.time()-t1))
    #
    # t2 = time.time()
    # y = np.arange(1000)
    # for i in np.nditer(y):
    #     y[i] = 0
    #     ''
    # print("NUMPY:"+str(time.time()-t2))
    #
    # t3 = time.time()
    # z = np.arange(1000)
    # index = 0
    # for i in z:
    #     z[index] = 0
    #     index+=1
    #     ''
    # print("NUMPY2:" + str(time.time() - t3))
    #
    # print(z)
    #
    # for i in x:
    #     print(i)

    #Loops through each of the new image and
    # for x in np.nditer(newimage, op_flags=['readwrite']):
    #     newIntensityValue = 0
    #
    #
    #
    #
    #     x[...] = newIntensityValue;
    #     num_pixels+=1

    #f = np.array([[2,4,43,5],[3,54,35,5]])
    #print(f.item(1,2))

    # Kernels
    MEAN_KERNEL = np.full((3, 3), 1)
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    SOBEL_Y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    SOBEL_X = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    BIG_GAUSSIAN_KERNEL = np.array([
        [1,1,1,1,1],
        [1,2,2,2,1],
        [1,2,5,2,1],
        [1,2,2,2,1],
        [1,1,1,1,1]
    ])

    BLURRY_KERNEL = np.array([
        [-0.5, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [-5435, 0, 2000, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, -4000]
    ])

    SHARPEN_IMAGE = np.array([
        [-0.5, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 7000, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, -4000]
    ])

    SHARPEN_IMAGE2 = np.array([
        [-0.5, 1, 1, 1, -5443],
        [1, 0, 44, 0, -5435],
        [1, 0, 7000, 0, -5454],
        [1, 0, 44, 0, 1],
        [0, 1, 1, 1, 0]
    ])

    SHARPEN_IMAGE3 = np.array([
        [-10, 1, 1, 1, 0],
        [1, 0, 44, 0, -10],
        [-10, 0, -5, 0, -10],
        [1, 0, 44, 0, 1],
        [0, 1, 1, 1, -10]
    ])

    selectedImagePath = getImageFromListOfImages(listOfImages,'nz1')
    print(selectedImagePath)

    wantGrayImage = False

    oldimage = getImageArray(selectedImagePath,wantGrayImage)
    #oldimage = Expand(Convolve(oldimage,GAUSSIAN_KERNEL),0.5)
    displayImageGivenArray(oldimage)
    print("HEIGHT:"+str(oldimage.shape[0])+","+"WIDTH:"+str(oldimage.shape[1]))

    newimage = Convolve(oldimage,SHARPEN_IMAGE3)
    #newimage = Convolve(oldimage,SOBEL_Y)
    displayImageGivenArray(newimage)
    print("Finished")
    #print(newimage)
    #print(newimage)



    #displayImageGivenArray(Expand(newimage,2))

    #displayImageGivenArray(Reduce(oldimage))

    #GaussianPyrmaid(oldimage,5)

main()