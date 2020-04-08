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




def displayImageGivenArray(numpyimagearr, windowTitle:str='image'):
    cv2.imshow(windowTitle, numpyimagearr)
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
                # if (summedKernelBlueValues >= 255):
                #     summedKernelBlueValues = 255
                # if (summedKernelBlueValues <= 0):
                #     summedKernelBlueValues = 0
                #
                # if (summedKernelGreenValues >= 255):
                #     summedKernelGreenValues = 255
                # if (summedKernelGreenValues <= 0):
                #     summedKernelGreenValues = 0
                #
                # if (summedKernelRedValues >= 255):
                #     summedKernelRedValues = 255
                # if (summedKernelRedValues <= 0):
                #     summedKernelRedValues = 0

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
    GAUSSIAN_KERNEL_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])

    curr_image = Convolve(I, GAUSSIAN_KERNEL)
    new_image = Scale(curr_image,factor)#Scales down image

    return new_image

def Scale(I,scaleFactor:int):
    new_width = int(I.shape[1]*scaleFactor)
    new_height = int(I.shape[0]*scaleFactor)

    print("NEW WIDTH:"+str(new_width)+",NEW HEIGHT:"+str(new_height))

    new_image = cv2.resize(I, (new_width,new_height))
    return new_image

def ScaleImage1ToImage2(image1,image2):
    print(image1)
    print(image2)

    newimage1 = None
    if(image1.shape[0]!=image2.shape[0] or image1.shape[1]!=image2.shape[1]):
        newimage1 = cv2.resize(image1, (image2.shape[1],image2.shape[0]))
        return newimage1
    else:
        return image1

def ScaleByGivenDimensions(I,dim):
    new_image = cv2.resize(I, (dim[0],dim[1]))
    return new_image

'''
takes image I as input and outputs copy of image expanded,
twice the width and height of the input. '''
def Expand(I,otherImage):
    return ScaleImage1ToImage2(I,otherImage)


'''
takes in an image I,
takes in N (int) is the no. of levels. 
'''
def GaussianPyramid(I,N:int, display:bool=False):
    curr_image = I
    gaussianpyramid = np.arange(N,dtype=np.ndarray)


    
    for i in range(0,N):
        try:
            gaussianpyramid.itemset(i,curr_image)
            curr_image = Reduce(curr_image)

        except:
            ''


    #print(gaussianpyramid)
    #print(len(gaussianpyramid))

    #Goes through each guassian blurred image and displays it

    if(display):
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
def LaplacianPyramids(I,N:int,displayLaplacian:bool=False, displayBothLaplacianAndGaussian:bool=False):
    print("Starting construction of LaplacianPyramid")
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    GAUSSIAN_KERNEL_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])

    print("Creating Gaussian pyramid for laplacian..")
    gaussianPyramid = GaussianPyramid(I,N)
    print("Finished Gaussian pyramid, starting laplacian pyramid")



    laplacianPyramid = list(range(N))
    NLevelLaplacian = gaussianPyramid.item(N-1)

    laplacianPyramid[N-1] = NLevelLaplacian
    for i in range(N-2,-1,-1):
        nextLevelGaussian = gaussianPyramid.item(i+1)
        currentGaussian = gaussianPyramid.item(i)
        expandedGaussian = Expand(nextLevelGaussian,currentGaussian)
        currentLaplacian = currentGaussian-expandedGaussian
        laplacianPyramid[i] = currentLaplacian



    print("Finished construction of Laplacian Pyramid, press a key to tap through images...")

    p = 0
    if(displayBothLaplacianAndGaussian):
        for image in gaussianPyramid:
            print("Displaying Gaussian Image Level " + str(p) + ":")
            displayImageGivenArray(image)
            p+=1

    if(displayLaplacian or displayBothLaplacianAndGaussian):
        for i in range(len(laplacianPyramid)-1,-1,-1):
            image = laplacianPyramid[i]
            print("Displaying Laplacian Image Level "+str(i)+":")
            displayImageGivenArray(image)


    return np.array(laplacianPyramid)



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

    for image in Ll:
        print("LL:")
        print(image)

    for i in range(N - 2, -1, -1):
        currentLaplacianImage = Ll.item(i)

        #Expand first and then blur because expanded image will have irregularities if not blurred
        #currentimage = Convolve(Expand(currentimage,currentLaplacianImage),GAUSSIAN_KERNEL)
        currentimage = Expand(currentimage, currentLaplacianImage)
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


def cropAndMaskGivenImages(foregroundImagePath,backgroundImagePath, wantGrayImage:bool=False):
    MEAN_KERNEL = np.full((3, 3), 1)
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    GAUSSIAN_KERNEL_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])

    circles = []
    def mouse_drawing(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print("Left click")
            circles.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            #print("MOVING")
            circles.append((x, y))

    foregroundImage= getImageArray(foregroundImagePath,wantGrayImage)
    foregroundImage_copy = foregroundImage.copy()

    backgroundImage= getImageArray(backgroundImagePath, wantGrayImage)
    backgroundImage_copy = backgroundImage.copy()
    print(circles)
    circles = []



    while True:
        imagewithDrawings = foregroundImage_copy.copy()
        for center_position in circles:
            cv2.circle(imagewithDrawings, center_position, 2, (0, 0, 255), -1)

        cv2.namedWindow("Foreground Image")
        cv2.setMouseCallback("Foreground Image", mouse_drawing)
        cv2.imshow("Foreground Image", imagewithDrawings)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            return
        elif key == ord("c"):
            cv2.destroyAllWindows()
            break
        elif key == ord("r"):
            foregroundImage_copy = foregroundImage_copy.copy() #clears out current image from any drawings
            circles = []


    # create a mask with white pixels
    mask = np.ones_like(foregroundImage_copy)
    print(mask)
    mask.fill(0)
    print(mask)

    cv2.polylines(mask, np.array([circles]),True, [255,255,255],1)
    outlinepoints = np.where(np.all(mask == [255,255,255], axis=-1))
    outlinepoints = zip(outlinepoints[0],outlinepoints[1])

    print(outlinepoints)
    cv2.fillPoly(mask, np.array([circles]), [255,255,255])

    print("Mask:")
    displayImageGivenArray(mask,windowTitle='Mask')


    cropped_image  = cv2.bitwise_and(mask, foregroundImage_copy.copy())


    NLevels=1

    foregroundImage_copy=ScaleImage1ToImage2(foregroundImage_copy,backgroundImage_copy)
    mask = ScaleImage1ToImage2(mask,backgroundImage_copy)


    FORELAPLACIAN = LaplacianPyramids(foregroundImage_copy, NLevels)
    BACKLAPLACIAN = LaplacianPyramids(backgroundImage_copy, NLevels)

    #Reconstruct(FORELAPLACIAN,NLevels)
    #Reconstruct(BACKLAPLACIAN,NLevels)

    MASKGAUSSIAN = GaussianPyramid(mask,NLevels,True)


    BLENDEDLAPLACIAN = []

    for i in range(NLevels-1,-1,-1):
        print("LEVEL:"+str(i))
        currentFore = np.float32(FORELAPLACIAN[i])
        print("CURRENT FORE:")
        #print(currentFore)
        #displayImageGivenArray(currentFore)


        currentBack = np.float32(BACKLAPLACIAN[i])
        print("CURRENT BACK:")
        #print(currentBack)
        #displayImageGivenArray(currentBack)


        currentMaskGaussian = np.float32(MASKGAUSSIAN[i])
        print("CURRENT MASK:")

        #print(currentMaskGaussian)

        print("WEIGHTS MASK GAUSSIAN")
        weightsMaskGaussian = currentMaskGaussian/255.0
        #print(weightsMaskGaussian)

        print("INVERTED MASK GAUSSIAN")
        invertedWeightsMaskGaussian = (255.0-currentMaskGaussian)/255.0
        #print(invertedWeightsMaskGaussian)

        np.set_printoptions(threshold=1000)

        #displayImageGivenArray(currentMaskGaussian)


        print("CURRENT BLENDED")

        LeftHalfBlendedLaplacian = np.uint8(currentFore*(weightsMaskGaussian))
        print("LEFT BLENDED:")
        np.set_printoptions(threshold=np.inf)
        #print(LeftHalfBlendedLaplacian)
        #displayImageGivenArray(LeftHalfBlendedLaplacian,'Left half')


        RightHalfBlendedLaplacian = np.uint8(currentBack*(invertedWeightsMaskGaussian))
        print("RIGHT BLENDED:")
        #print(RightHalfBlendedLaplacian)
        #displayImageGivenArray(RightHalfBlendedLaplacian,'Right Half')
        currentBlendedLaplacian = np.uint8(LeftHalfBlendedLaplacian+RightHalfBlendedLaplacian)

        # for point in outlinepoints:
        #     print("Averaging the outline!")
        #     print(point)
        #     print(currentBlendedLaplacian[51][126])
        #     print(currentBlendedLaplacian.shape)
        #     currentBlendedPoint = currentBlendedLaplacian[point[0]][point[1]]
        #     currentBlue = currentBlendedPoint[0]
        #     currentGreen = currentBlendedPoint[1]
        #     currentRed = currentBlendedPoint[2]
        #
        #     newBlendedPoint = [currentBlue/2,currentGreen/2,currentRed/2]
        #     currentBlendedLaplacian[point[0]][point[1]] = newBlendedPoint


        print("CURRENT BLENDED:")
        print(currentBlendedLaplacian)
        #displayImageGivenArray(currentBlendedLaplacian,"Current Blended:")




        BLENDEDLAPLACIAN = [currentBlendedLaplacian]+BLENDEDLAPLACIAN


    combinedimage = Reconstruct(np.array(BLENDEDLAPLACIAN), NLevels)

    # for point in outlinepoints:
    #     combinedimage[point[0]][point[1]] = [0,0,255]

    cv2.imshow("Combined image:",combinedimage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def cropAndCombineGivenImages(listofimagepaths:[str], wantGrayImage:bool=False):
    MEAN_KERNEL = np.full((3, 3), 1)
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    GAUSSIAN_KERNEL_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
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
        image = getImageArray(imagepath,wantGrayImage)
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

    NLevels=5
    combinedimage = croppedimages[0] #combined image starts with first cropped image
    print(combinedimage)
    for i in range(1,len(croppedimages)):
        if (croppedimages[i].shape[0] != combinedimage.shape[0] or croppedimages[i].shape[1] != combinedimage.shape[1]):
            croppedimages[i]= ScaleByGivenDimensions(croppedimages[i], (combinedimage.shape[1], combinedimage.shape[0]))

        else:
            ''








        combinedimage = cv2.addWeighted(combinedimage,0.5,croppedimages[i],0.5,0)


    cv2.imshow("Combined image:",combinedimage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def findCorrespondingImage2Point(point,image1image2points:dict):
    #The image one points matrix: (used to find affine parameters)
    imageOnePoints_matrix = np.zeros((2*len(image1image2points),6))

    #The image two points for image one points: (used to find affine parameters)
    resultsTwoPoints_matrix = np.zeros((2*len(image1image2points),1))

    #Creates the 2d arrays for n amount of points
    for i in range(len(image1image2points)):
        imageOnePoints_matrix[i*2] = [image1image2points[i]['img1_x'],image1image2points[i]['img1_y'],0,0,1,0]
        imageOnePoints_matrix[(i*2)+1] = [0,0,image1image2points[i]['img1_x'],image1image2points[i]['img1_y'],0,1]
        
        resultsTwoPoints_matrix[i*2]= image1image2points[i]['img2_x']
        resultsTwoPoints_matrix[(i*2)+1]= image1image2points[i]['img2_y']

    m1m2m3m4txty_matrix = None

    #If the image1image2points contains more than 3 points then fixes overconstraint:
    if(3<len(image1image2points)):
        m1m2m3m4txty_matrix = (np.linalg.inv(imageOnePoints_matrix.T@imageOnePoints_matrix)@imageOnePoints_matrix.T)@resultsTwoPoints_matrix

    #If the image1image2points contain 3 points then multiply by the inverse matrix
    elif(len(image1image2points)==3):
        m1m2m3m4txty_matrix = np.linalg.inv(imageOnePoints_matrix)@resultsTwoPoints_matrix
    else:
        raise Exception("ERROR:The image1image2points are neither 3 points or more!")

    #Pulls out the transformation parameters:
    m1m2m3m4_matrix = np.array([
        [m1m2m3m4txty_matrix[0][0], m1m2m3m4txty_matrix[1][0]],
        [m1m2m3m4txty_matrix[2][0], m1m2m3m4txty_matrix[3][0]]])
    txty_matrix = np.array([
        [m1m2m3m4txty_matrix[4][0]],
        [m1m2m3m4txty_matrix[5][0]]
    ])

    #print("m1m2m3m4_matrix:")
    #print(m1m2m3m4_matrix)
    #print(txty_matrix.shape)


    #Converts image1 point into matrix equivalent:
    image1Point = np.array([
        [point[0]],
        [point[1]]
    ])

    #print("imageOnePoints_matrix:")
    #print(imageOnePoints_matrix)
    #print(image1Point)

    #Calculates the corresponding image 2 point by the transformation paramters and image 1 point:
    correspondingImage2Point = (m1m2m3m4_matrix@image1Point)+txty_matrix


    print("Old Point:[" + str(image1Point[0][0]) + "," + str(image1Point[1][0]) + "]")
    print("New Point:["+str(correspondingImage2Point[0][0])+","+str(correspondingImage2Point[1][0])+"]")

    return correspondingImage2Point


circles = []
def main():
    #PART1-------------------------------
    #Kernels
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

    GAUSSIAN_KERNEL_5x5 = np.array([
        [1,4,7,4,1],
        [4,16,26,16,4],
        [7,26,41,26,7],
        [4,16,26,16,4],
        [1,4,7,4,1]
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

    #Gets a list of imagepaths (Strings) to all images contained within ./input_images folder:
    listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR,True)  

    #Select the specified image from the list of images by giving it the name of the image:
    selectedImagePath = getImageFromListOfImages(listOfImages,'im1')
    print(selectedImagePath)

    #If wantGrayImage is false gets a colored image, otherwise gets a gray image
    wantGrayImage = False
    image = getImageArray(selectedImagePath,wantGrayImage)
    image_copy = image.copy()

    #print(np.zeros_like(image))
    print(cv2.bitwise_not(np.zeros_like(image))/0.5)


    #print("Started GAUSSIAN 5x5")
    #displayImageGivenArray(Convolve(image_copy,GAUSSIAN_KERNEL_5x5))

    #print("Started GAUSSIAN")
    #displayImageGivenArray(Convolve(image_copy,GAUSSIAN_KERNEL))
    #Reconstruct(LaplacianPyramids(image_copy,8))

    #Reconstruct(LaplacianPyramids(getImageArray(listOfImages[7],False),10,displayBothLaplacianAndGaussian=True))
    #Reconstruct(LaplacianPyramids(image_copy,5,True))
    cropAndMaskGivenImages(listOfImages[4],listOfImages[3],wantGrayImage)



    #displayImageGivenArray(oldimage)



    #PART2-------------------------------------------------------------
    #Gets corresponding images 1,2:
    mountainimage1 = getImageFromListOfImages(listOfImages, 'im1_1')
    mountainimage2 = getImageFromListOfImages(listOfImages, 'im1_2')

    #(These two set of points already contain points selected from cpselect)
    threeimage1image2points =[{'point_id': 1, 'img1_x': 209.89760003380235, 'img1_y': 81.64180081970682, 'img2_x': 64.54077407360467, 'img2_y': 99.30352389402975}, {'point_id': 2, 'img1_x': 304.71316601174635, 'img1_y': 169.95041619132132, 'img2_x': 154.70851818988467, 'img2_y': 193.18952549964092}, {'point_id': 3, 'img1_x': 246.15061055478094, 'img1_y': 210.8512485739638, 'img2_x': 93.35726961592093, 'img2_y': 233.16079350995062}]
    morethanthreeimage1image2points =[{'point_id': 1, 'img1_x': 209.89760003380235, 'img1_y': 81.64180081970682, 'img2_x': 64.54077407360467, 'img2_y': 99.30352389402975}, {'point_id': 2, 'img1_x': 304.71316601174635, 'img1_y': 169.95041619132132, 'img2_x': 154.70851818988467, 'img2_y': 193.18952549964092}, {'point_id': 3, 'img1_x': 246.15061055478094, 'img1_y': 210.8512485739638, 'img2_x': 93.35726961592093, 'img2_y': 233.16079350995062}, {'point_id': 4, 'img1_x': 201.5315206828073, 'img1_y': 150.42956437233283, 'img2_x': 51.52687286094567, 'img2_y': 170.8799805636541}, {'point_id': 5, 'img1_x': 393.0217813833608, 'img1_y': 47.247919043393836, 'img2_x': 243.01713356149924, 'img2_y': 78.85310770270848}]

    selectedPoints = threeimage1image2points
    #print(selectedPoints)

    #(Uncomment below to get points manually using cpselect)
    #selectedPoints = cpselect(mountainimage1,mountainimage2)

    #Test point (verified to be about the same feature in image 1,2)
    point4_image1 = [morethanthreeimage1image2points[4]['img1_x'],morethanthreeimage1image2points[4]['img1_y']]
    point4_image2 = [morethanthreeimage1image2points[4]['img2_x'],morethanthreeimage1image2points[4]['img2_y']]

    findCorrespondingImage2Point(point4_image1, selectedPoints)
    print("Actual:"+str(point4_image2) + '\n')

    findCorrespondingImage2Point((366.06,107.67), selectedPoints)
    print("Actual:216.06, 135.56"+ '\n')
    #cpselect(mountainimage1, mountainimage2)
    #366.06,107.67  216.06, 135.56

    findCorrespondingImage2Point((361.42, 133.7), selectedPoints)
    print("Actual:211.41, 159.73" + '\n')
    #361.42, 133.7, 211.41, 159.73
main()
