import numpy as np
import cv2
import time
import os
import math
from matplotlib import pyplot
from cpselect.cpselect import cpselect
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
        #print("THIS IS A GRAYSCALE IMAGE")
    else:
        isColorImage = True
        #print("THIS IS A COLOR IMAGE")

    #print(newimage)

    #print(image_width,image_height)

    centerOfKernel = -1

    kerneldimensions = H.shape
    kernel_height = kerneldimensions[0]
    kernel_width = kerneldimensions[1]

    middleRow = math.floor(kernel_height/2)
    middleColumn = math.floor(kernel_width/2)
    kernelTotal = np.sum(a=H)
    #print("KERNEL TOTAL:"+str(kernelTotal))
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
def Reduce(I,factor:int=0.5):
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])


    new_image = Convolve(I,GAUSSIAN_KERNEL) #Blurs image using Gaussian Blur kernel
    new_image = Scale(new_image,factor)#Scales down image

    return new_image

def Scale(I,scaleFactor:int):
    new_width = int(I.shape[1]*scaleFactor)
    new_height = int(I.shape[0]*scaleFactor)

    print("NEW WIDTH:"+str(new_width)+",NEW HEIGHT:"+str(new_height))

    new_image = cv2.resize(I, (new_width,new_height))
    return new_image

def ScaleByGivenDimensions(I,dim):
    new_image = cv2.resize(I, (dim[0],dim[1]))
    return new_image

'''
takes image I as input and outputs copy of image expanded,
twice the width and height of the input. '''
def Expand(I,otherImage):
    return ScaleByGivenDimensions(I,(otherImage.shape[1],otherImage.shape[0]))


'''
takes in an image I,
takes in N (int) is the no. of levels. 
'''
def GaussianPyramid(I,N:int):
    curr_image = I
    gaussianpyramid = np.arange(N,dtype=np.ndarray)
    gaussianpyramid.itemset(0,curr_image)

    #Creates the gaussian blurred images and places them in guassianpryamid variable
    #Adds N-1 gaussian blurs to the array since first level is the original image
    for i in range(1,N):
        try:
            curr_image = Reduce(curr_image)
            gaussianpyramid.itemset(i,curr_image)
            #print(i)
        except:
            ''
            #print("COULD NOT REDUCE FURTHER")
            #print(i)
            #gaussianpyramid.itemset(i,np.array([]))


    #print(gaussianpyramid)
    #print(len(gaussianpyramid))

    #Goes through each guassian blurred image and displays it
    Level = 0
    for i in gaussianpyramid:
        print("Level:" + str(Level))
        try:
            displayImageGivenArray(i)
        except:
            print("Could not display this Guassian Level:"+str(Level)+" image!(perhaps too small?)")

        Level+=1

    return gaussianpyramid

'''
that produces n level Laplacian pyramid of I.
'''
def LaplacianPyramids(I,N:int):
    print("Starting construction of LaplacianPyramid")
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    BIG_GAUSSIAN_KERNEL = np.array([
        [1,1,1,1,1],
        [1,2,2,2,1],
        [1,2,5,2,1],
        [1,2,2,2,1],
        [1,1,1,1,1]
    ])

    listofgaussianimages = []
    listoflaplacianimages = []

    currentImage = I
    for i in range(0,N):
        currentGaussianImage = Convolve(currentImage,GAUSSIAN_KERNEL)
        listofgaussianimages.append(currentGaussianImage)

        #displayImageGivenArray(currentGaussianImage)


        currentLaplacianImage = currentImage-currentGaussianImage
        listoflaplacianimages.append(currentLaplacianImage)
        #displayImageGivenArray(currentLaplacianImage)

        currentImage = Reduce(currentImage)

    listoflaplacianimages[len(listoflaplacianimages)-1]=currentImage
    print("Finished construction of Laplacian Pyramid, press a key to tap through images...")

    p = 0
    for image in listofgaussianimages:
        print("Displaying Gaussian Image Level " + str(p) + ":")
        displayImageGivenArray(image)
        p+=1

    for i in range(len(listoflaplacianimages)-1,-1,-1):
        image = listoflaplacianimages[i]
        print("Displaying Laplacian Image Level "+str(i)+":")
        displayImageGivenArray(image)


    return np.array(listoflaplacianimages)



    ''

'''
function which collapses the Laplacian pyramid LI of nlevels 
to generate the original image. Report the error in reconstruction
using image difference. 
'''
def Reconstruct(Ll,N:int = -1):
    lenLaplacian = len(Ll)
    if(N ==-1):
        N = lenLaplacian
    print("Starting reconstruction, please wait...")
    originalimage = None
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    listofreconstructedimages = np.arange(lenLaplacian,dtype=np.ndarray)
    currentimage = Ll[lenLaplacian-1]
    listofreconstructedimages.itemset(lenLaplacian-1,currentimage)

    for i in range(N - 2, -1, -1):
        currentLaplacianImage = Ll.item(i)

        #Expand first and then blur because expanded image will have irregularities if not blurred
        currentimage = Convolve(Expand(currentimage,currentLaplacianImage),GAUSSIAN_KERNEL)
        #currentimage = Expand(currentimage, currentLaplacianImage)
        currentimage = currentimage+currentLaplacianImage
        #displayImageGivenArray(currentimage)
        listofreconstructedimages.itemset(i,currentimage)

    originalimage = currentimage
    print("Finished reconstruction, press a key to begin displaying images..")

    foundFirstImage = False
    for i in range(len(listofreconstructedimages)-1,-1,-1):
        image = listofreconstructedimages.item(i)
        if(isinstance(image,np.ndarray) and foundFirstImage==False):
            print("Displaying smallest image at Level "+str(i)+":")
            foundFirstImage = True
        elif(isinstance(image,np.ndarray)):
            print("Displaying reconstructed image Level "+str(i)+":")
        else:
            continue


        displayImageGivenArray(image)

    return originalimage


# refPt = []
# cropping = False
# def click_and_crop(image,event,x,y):
#     # grab references to the global variables
#     global refPt, cropping
#     # if the left mouse button was clicked, record the starting
#     # (x, y) coordinates and indicate that cropping is being
#     # performed
#     if event == cv2.EVENT_LBUTTONDOWN:
#         refPt = [(x, y)]
#         cropping = True
#     # check to see if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         # record the ending (x, y) coordinates and indicate that
#         # the cropping operation is finished
#         refPt.append((x, y))
#         cropping = False
#         # draw a rectangle around the region of interest
#         cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
#         cv2.imshow("image", image)
#         # load the image, clone it, and setup the mouse callback function
#         clone = image.copy()
#         cv2.namedWindow("image")
#         cv2.setMouseCallback("image", click_and_crop)
#         # keep looping until the 'q' key is pressed
#         while True:
#             # display the image and wait for a keypress
#             cv2.imshow("image", image)
#             key = cv2.waitKey(1) & 0xFF
#             # if the 'r' key is pressed, reset the cropping region
#             if key == ord("r"):
#                 image = clone.copy()
#             # if the 'c' key is pressed, break from the loop
#             elif key == ord("c"):
#                 break
#         # if there are two reference points, then crop the region of interest
#         # from teh image and display it
#         if len(refPt) == 2:
#             roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
#             cv2.imshow("ROI", roi)
#             cv2.waitKey(0)
#         # close all open windows
#         cv2.destroyAllWindows()

def cropAndBlendGivenImages(listofimagepaths:[str]):
    MEAN_KERNEL = np.full((3, 3), 1)
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    circles = []
    def mouse_drawing(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print("Left click")
            circles.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            #print("MOVING")
            circles.append((x, y))

    croppedimages = []
    imageid = 0
    largestimagefound = None
    for imagepath in listofimagepaths:
        selectedCropEntireImage = False
        image = getImageArray(imagepath,False)
        image_copy = image.copy()
        print(circles)
        circles = []


        if(isinstance(largestimagefound,np.ndarray)==False):
            largestimagefound = image_copy
        elif((largestimagefound.shape[0]*largestimagefound.shape[1])<(image_copy.shape[0]*image_copy.shape[1])):
            largestimagefound = image_copy
            print("Found a larger image!")



        while True:
            imagewithDrawings = image_copy.copy()
            for center_position in circles:
                cv2.circle(imagewithDrawings, center_position, 2, (0, 0, 255), -1)

            cv2.namedWindow("Image"+str(imageid))
            cv2.setMouseCallback("Image"+str(imageid), mouse_drawing)
            cv2.imshow("Image"+str(imageid), imagewithDrawings)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                return
            elif key == ord("c"):
                cv2.destroyAllWindows()
                break
            elif key == ord("r"):
                image_copy = image.copy() #clears out current image from any drawings
                circles = []
            elif key == ord("a"):
                print("Selected entire region")
                image_copy = image.copy()
                croppedimages.append(image_copy)
                selectedCropEntireImage = True
                cv2.destroyAllWindows()
                break

        imageid += 1
        if(selectedCropEntireImage):
            continue
        else:
            # create a mask with white pixels
            mask = np.zeros_like(image_copy)
            print(mask)
            mask.fill(0)
            print(mask)

            cv2.fillPoly(mask, np.array([circles]), (255, 255, 255))
            print(mask)
            newimage = image.copy()
            masked_image = cv2.bitwise_and(mask, image_copy)
            croppedimages.append(masked_image)

    print(imageid)
    combinedimage = np.zeros_like(largestimagefound)
    # for i in range(len(croppedimages)):
    #     if(croppedimages[i].shape[0]!=combinedimage.shape[0] or croppedimages[i].shape[1]!=combinedimage.shape[1]):
    #         combinedimage = combinedimage+(Convolve(ScaleByGivenDimensions(croppedimages[i],(combinedimage.shape[1],combinedimage.shape[0])),GAUSSIAN_KERNEL))*.5
    #     else:
    #         combinedimage=combinedimage+croppedimages[i]
    #
    #     cv2.imshow("Cropped Image:" + str(i), croppedimages[i])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    NLevels=5
    combinedimage = croppedimages[0] #combined image starts with first cropped image
    print(combinedimage)
    for i in range(1,len(croppedimages)):
        if (croppedimages[i].shape[0] != combinedimage.shape[0] or croppedimages[i].shape[1] != combinedimage.shape[1]):
            croppedimages[i]= (
                Convolve(ScaleByGivenDimensions(croppedimages[i], (combinedimage.shape[1], combinedimage.shape[0])),
                         GAUSSIAN_KERNEL))
        else:
            ''




        LA = LaplacianPyramids(combinedimage,NLevels)
        LB = LaplacianPyramids(croppedimages[i],NLevels)
        LS = LA+LB
        combinedimage = Convolve(Reconstruct(LS,NLevels),GAUSSIAN_KERNEL)

    cv2.imshow("Combined image:",combinedimage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()



def solveForUVUsingTransformParameters(point, m1m2m3m4, txty,p1a,p2a,p3a):
    image1pointsmatrix = np.array([
        [p1a[0],p1a[1],1,0,0,0],
        [0,0,0,p1a[0],p1a[1],1],
        [p2a[0],p2a[1],1,0,0,0],
        [0,0,0,p2a[0],p2a[1],1],
        [p3a[0],p3a[1],1,0,0,0],
        [0,0,0,p3a[0],p3a[1],1]
    ])
    if((isinstance(point,list)==False) and isinstance(point,tuple)==False):
        raise Exception('ERROR, input a tuple or list representing a x,y point')
        return

    if(isinstance(point,list) and len(point)>2):
        raise Exception('ERROR, the length of list is greater than 2!, should be a x,y point')
        return

    if ((isinstance(txty, list) == False) and isinstance(txty, tuple) == False):
        raise Exception('ERROR, input a tuple or list representing a x,y point')
        return

    if (isinstance(txty, list) and len(txty) > 2):
        raise Exception('ERROR, the length of list is greater than 2!, should be a x,y point')
        return

    if ((isinstance(m1m2m3m4, list) == False) and isinstance(m1m2m3m4, tuple) == False):
        raise Exception('ERROR, input a tuple or list representing a x,y point')
        return

    x = point[0]
    y = point[1]
    xy_matrix = np.array([
        [x,y,0,0,1,0],
        [0,0,x,y,0,1]
    ])

    m1 = m1m2m3m4[0]
    m2 = m1m2m3m4[1]
    m3 = m1m2m3m4[2]
    m4 = m1m2m3m4[3]
    tx = txty[0]
    ty = txty[1]
    m1m2m3m4txty_matrix = np.array([
        [m1],
        [m2],
        [m3],
        [m4],
        [tx],
        [ty]
    ])

    uv_matrix = np.dot(image1pointsmatrix,m1m2m3m4txty_matrix)
    print(uv_matrix)
    return uv_matrix





circles = []
def main():
    listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR,True)

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

    RAND1 = np.array([
        [25, 0, 100],
        [0, -255, 25],
        [100, 0, 25]
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

    selectedImagePath = getImageFromListOfImages(listOfImages,'im1')
    print(selectedImagePath)

    wantGrayImage = False  #If false gets a colored image
    oldimage = getImageArray(selectedImagePath,wantGrayImage)
    oldimagecopy = oldimage.copy()
    #displayImageGivenArray(oldimage)
    #Reconstruct(LaplacianPyramids(oldimage,8))



    print("WORKING")


    #cropAndBlendGivenImages(listOfImages[3:6])

    # imageA = getImageArray(listOfImages[0],False)
    # imageB = getImageArray(listOfImages[4],False)
    # imageB=ScaleByGivenDimensions(imageB, (imageA.shape[1], imageA.shape[0]))
    #
    # LA = LaplacianPyramids(imageA,5)
    # LB = LaplacianPyramids(imageB,5)

    cropregion = None
    #cropregion[0:60] = [[[255,255,255]]]
    #print(cropregion)

    #displayImageGivenArray(oldimage)
    mountainimage1 = getImageFromListOfImages(listOfImages, 'im1_1')
    mountainimage2 = getImageFromListOfImages(listOfImages, 'im1_2')
    #controlpointlist = cpselect(mountainimage1,mountainimage2)

    #print(controlpointlist)

    #three more points:
    #[{'point_id': 1, 'img1_x': 230.3480162251236, 'img1_y': 204.3442979676343, 'img2_x': 77.55467528626366, 'img2_y': 226.65384290362113}, {'point_id': 2, 'img1_x': 225.70019436345967, 'img1_y': 155.07738623399678, 'img2_x': 75.6955465415981, 'img2_y': 178.31649554231637}, {'point_id': 3, 'img1_x': 299.13577977774963, 'img1_y': 178.31649554231637, 'img2_x': 150.06069632822084, 'img2_y': 202.48516922296875}]


    selectedPoints = [{'point_id': 1, 'img1_x': 212.6862931508007, 'img1_y': 82.5713651920396, 'img2_x': 66.39990281827022, 'img2_y': 104.88091012802641}, {'point_id': 2, 'img1_x': 395.8104745003592, 'img1_y': 45.388790298728225, 'img2_x': 283.91796594414166, 'img2_y': 49.10704778805939}, {'point_id': 3, 'img1_x': 238.71409557611867, 'img1_y': 210.8512485739638, 'img2_x': 85.92075463725871, 'img2_y': 229.44253602061946},
                      {'point_id': 3, 'img1_x': 230.3480162251236, 'img1_y': 204.3442979676343,
                       'img2_x': 77.55467528626366, 'img2_y': 226.65384290362113},
                      {'point_id': 4, 'img1_x': 225.70019436345967, 'img1_y': 155.07738623399678,
                       'img2_x': 75.6955465415981, 'img2_y': 178.31649554231637},
                      {'point_id': 5, 'img1_x': 299.13577977774963, 'img1_y': 178.31649554231637,
                       'img2_x': 150.06069632822084, 'img2_y': 202.48516922296875}]
    print(selectedPoints)

    p1a = np.float32([selectedPoints[0]['img1_x'],selectedPoints[0]['img1_y']])
    p1b = np.float32([selectedPoints[0]['img2_x'],selectedPoints[0]['img2_y']])

    p2a = np.float32([selectedPoints[1]['img1_x'],selectedPoints[1]['img1_y']])
    p2b = np.float32([selectedPoints[1]['img2_x'], selectedPoints[1]['img2_y']])

    p3a = np.float32([selectedPoints[2]['img1_x'], selectedPoints[2]['img1_y']])
    p3b = np.float32([selectedPoints[2]['img2_x'], selectedPoints[2]['img2_y']])

    p4a = np.float32([selectedPoints[3]['img1_x'],selectedPoints[0]['img1_y']])
    p4b = np.float32([selectedPoints[3]['img2_x'],selectedPoints[0]['img2_y']])

    p5a = np.float32([selectedPoints[4]['img1_x'],selectedPoints[1]['img1_y']])
    p5b = np.float32([selectedPoints[4]['img2_x'], selectedPoints[1]['img2_y']])

    p6a = np.float32([selectedPoints[5]['img1_x'], selectedPoints[2]['img1_y']])
    p6b = np.float32([selectedPoints[5]['img2_x'], selectedPoints[2]['img2_y']])

    #print(p3a,p3b)

    #Contains only three points:
    image2pointsmatrix = np.array([
        [p1b[0]],
        [p1b[1]],
        [p2b[0]],
        [p2b[1]],
        [p3b[0]],
        [p3b[1]],
    ])

    #Contains 6 total points:
    # image2pointsmatrix = np.array([
    #     [p1b[0]],
    #     [p1b[1]],
    #     [p2b[0]],
    #     [p2b[1]],
    #     [p3b[0]],
    #     [p3b[1]],
    #     [p4b[0]],
    #     [p4b[1]],
    #     [p5b[0]],
    #     [p5b[1]],
    #     [p6b[0]],
    #     [p6b[1]],
    # ])

    print(image2pointsmatrix)

    #Contains only 3 points:
    image1pointsmatrix = np.array([
        [p1a[0],p1a[1],1,0,0,0],
        [0,0,0,p1a[0],p1a[1],1],
        [p2a[0],p2a[1],1,0,0,0],
        [0,0,0,p2a[0],p2a[1],1],
        [p3a[0],p3a[1],1,0,0,0],
        [0,0,0,p3a[0],p3a[1],1]
    ])

    #Contains 6 points in total:
    # image1pointsmatrix = np.array([
    #     [p1a[0], p1a[1], 1, 0, 0, 0,0,0,0,0,0,0],
    #     [0, 0, 0, p1a[0], p1a[1], 1,0,0,0,0,0,0],
    #     [p2a[0], p2a[1], 1, 0, 0, 0,0,0,0,0,0,0],
    #     [0, 0, 0, p2a[0], p2a[1], 1,0,0,0,0,0,0],
    #     [p3a[0], p3a[1], 1, 0, 0, 0,0,0,0,0,0,0],
    #     [0, 0, 0, p3a[0], p3a[1], 1,0,0,0,0,0,0],
    #     [p4a[0], p4a[1], 1, 0, 0, 0,0,0,0,0,0,0],
    #     [0, 0, 0, p4a[0], p4a[1], 1,0,0,0,0,0,0],
    #     [p5a[0], p5a[1], 1, 0, 0, 0,0,0,0,0,0,0],
    #     [0, 0, 0, p5a[0], p5a[1], 1,0,0,0,0,0,0],
    #     [p6a[0], p6a[1], 1, 0, 0, 0,0,0,0,0,0,0],
    #     [0, 0, 0, p6a[0], p6a[1], 1,0,0,0,0,0,0]
    # ])

    print("IMAGE ONE:")
    print(image1pointsmatrix)
    print(image1pointsmatrix.shape)

    #Used for 3 points only:
    m0,m1,m2,m3,m4,m5 = None,None,None,None,None,None

    #Used for 6 points:
    #m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11 = None,None,None,None,None,None,None,None,None,None,None,None

    affineparameters = np.array([
        [m0],
        [m1],
        [m2],
        [m3],
        [m4],
        [m5],
    ])

    #Used for 6 points:
    # affineparameters = np.array([
    #     [m0],
    #     [m1],
    #     [m2],
    #     [m3],
    #     [m4],
    #     [m5],
    #     [m6],
    #     [m7],
    #     [m8],
    #     [m9],
    #     [m10],
    #     [m11]
    # ])

    print("AFFINEPARAMETERS:")
    #print(affineparameters)

    np.linalg.inv(image1pointsmatrix)
    affineparameters = np.dot(np.linalg.inv(image1pointsmatrix),image2pointsmatrix)
    #print(affineparameters)
    print("INDEXING:" + str(affineparameters[0][0]))

    m_matrix = np.asarray([
        [affineparameters[1][0],affineparameters[2][0]],
        [affineparameters[3][0], affineparameters[4][0]]
        ])
    print("M MATRIX:")
    print(m_matrix)
    print(m_matrix.shape)
    print("XY MATRIX:")
    p1a_matrix = np.array([
        [p1a[0]],
        [p1a[1]]

    ])
    p1b_matrix = np.array([
        [p1b[0]],
        [p1b[1]]
    ])
    print(p1a_matrix.shape)

    txty_matrix = p1b_matrix-np.dot(m_matrix, p1a_matrix)

    print("TXTY:")
    print(txty_matrix.shape)
    print(txty_matrix)
    print("--------")

    print()

    supposedlyUV = np.add(np.dot(m_matrix, p1a_matrix),txty_matrix)
    print(p1b)
    print(supposedlyUV)

    testpointInImage2_matrix = np.array([
        [],
        []
    ])

    #[{'point_id': 1, 'img1_x': 301.92447289474796, 'img1_y': 170.8799805636541, 'img2_x': 152.84938944521912, 'img2_y': 194.1190898719737}]
    solveForUVUsingTransformParameters((301.92447289474796,170.8799805636541), (m_matrix[0][0],m_matrix[0][1],m_matrix[1][0],m_matrix[1][1]), (txty_matrix[0][0],txty_matrix[1][0]),p1a,p2a,p3a)
    newpoint = np.add(np.dot(m_matrix, np.array([
        [p3a[0]],
        [p3a[1]]
    ])), txty_matrix)
    print(newpoint)

    print(cv2.getAffineTransform(np.array([p1a,p2a,p3a]),np.array([p1b,p2b,p3b])))





main()
