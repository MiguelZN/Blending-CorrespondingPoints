import numpy as np
import cv2
import os
import math
import tkinter
from tkinter import filedialog


IMAGEDIR = './input_images/'

'''
Miguel Zavala
4/8/20
CISC442:Computer Vision
Project 1: blending, associative points
'''

#Searches a list of strings and looks for specified image name (string)
def getImageFromListOfImages(listofimages, want):
    for imagepath in listofimages:
        print(imagepath)
        if(want in imagepath):
            return imagepath

    raise Exception('Could not find image with this name!')


#Takes a path directory (string) and checks for all images in that directory
#Returns a list of image paths (list of strings)
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

#Takes in an imagepath (string) and displays the image
def displayImageGivenPath(imagepath:str, wantGrayImage=False):
    img = getImageArray(imagepath,wantGrayImage)
    cv2.imshow('image', img)
    cv2.waitKey(0) #waits for user to type in a key
    cv2.destroyAllWindows()

#Takes in an image (np array) and displays the image
def displayImageGivenArray(numpyimagearr, windowTitle:str='image', waitKey:int=0):
    cv2.imshow(windowTitle, numpyimagearr)
    cv2.waitKey(waitKey)  # waits for user to type in a key
    cv2.destroyAllWindows()

#Takes in an image path (string) and returns the image as a np array
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
                # if (summedKernelIntensityValues >= 255):
                #     summedKernelIntensityValues = 255
                # if (summedKernelIntensityValues <= 0):
                #     summedKernelIntensityValues = 0

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

#Scales a given image by a scaleFactor
def Scale(I,scaleFactor):
    new_width = int(I.shape[1]*scaleFactor)
    new_height = int(I.shape[0]*scaleFactor)

    print("NEW WIDTH:"+str(new_width)+",NEW HEIGHT:"+str(new_height))

    new_image = cv2.resize(I, (new_width,new_height))
    return new_image

#Takes in two images (np arrays) and scales image 1 to the exact dimensions of image 2
#(did this because sometimes dividing by 2 would result in different sized matrices due to
#rounding or ceiling or flooring by python)
def ScaleImage1ToImage2(image1,image2):
    # print(image1)
    # print(image2)

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
    GAUSSIAN_KERNEL = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    return Convolve(ScaleImage1ToImage2(I,otherImage),GAUSSIAN_KERNEL)


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

    print("Finished construction of Laplacian Pyramid..")
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



#Method used for blending two images
#Allows user to manually draw their own mask onto the foreground image
def cropAndMaskGivenImages(foregroundImagePath,backgroundImagePath, wantGrayImage:bool=False, NLevels:int=1):
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

    #Method setting up the mouse listeners to detect a user's drawing actions
    #(Example holding down left mouse and moving which draws circle points on the image)
    def mouse_drawing(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print("Left click")
            circles.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            #print("MOVING")
            circles.append((x, y))

    #Converts the imagepaths into np array images:
    foregroundImage= getImageArray(foregroundImagePath,wantGrayImage)
    foregroundImage_copy = foregroundImage.copy()
    backgroundImage= getImageArray(backgroundImagePath, wantGrayImage)
    backgroundImage_copy = backgroundImage.copy()


    #This holds all of the user drawn points when selecting their mask
    circles = []

    print("->Left Click or hold down Left Click and move to draw onto the image1 to create your own mask.\n"
          "->When finished, press key 'C' to select the mask.\n"
          "->Press key 'R' to reset your drawn points.\n"
          "->Press key 'U' to undo your last drawn point")

    #Allows the user to draw on the Foreground image and manually make their own mask
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
            print("Removing all drawn points...")
            foregroundImage_copy = foregroundImage_copy.copy() #clears out current image from any drawings
            circles = []
        elif key == ord("u"):
            print("undo successful..")
            circles = circles[0:len(circles)-1]


    #Creates a mask with black pixels
    mask = np.ones_like(foregroundImage_copy)
    mask.fill(0)

    #Gets the mask outline (does not fully work)
    #cv2.polylines(mask, np.array([circles]),True, [255,255,255],1)
    #outlinepoints = np.where(np.all(mask == [255,255,255], axis=-1))
    #outlinepoints = zip(outlinepoints[0],outlinepoints[1])
    #print(outlinepoints)

    cv2.fillPoly(mask, np.array([circles]), [255,255,255]) #Fills the mask area with white pixels (based on the points user selected)
    displayImageGivenArray(mask,windowTitle='Mask',waitKey=1000)

    #Gets cropped out image using bitmask (not needed)
    #cropped_image  = cv2.bitwise_and(mask, foregroundImage_copy.copy())

    foregroundImage_copy=ScaleImage1ToImage2(foregroundImage_copy,backgroundImage_copy)
    mask = ScaleImage1ToImage2(mask,backgroundImage_copy)

    #Creating the Pyramids (2 Laplacian, 1 Gaussian)
    FORELAPLACIAN = LaplacianPyramids(foregroundImage_copy, NLevels)
    BACKLAPLACIAN = LaplacianPyramids(backgroundImage_copy, NLevels)
    MASKGAUSSIAN = GaussianPyramid(mask,NLevels)

    #Holds the newly created Laplacian Pyramid
    BLENDEDLAPLACIAN = []
    for i in range(NLevels-1,-1,-1):
        currentFore = np.float32(FORELAPLACIAN[i]) #Current level foreground image of laplacian pyramid
        currentBack = np.float32(BACKLAPLACIAN[i]) #Current level background image of laplacian pyramid
        currentMaskGaussian = np.float32(MASKGAUSSIAN[i]) #Current level mask image of gaussian pyramid

        weightsMaskGaussian = currentMaskGaussian/255.0 #Creates weighted mask
        invertedWeightsMaskGaussian = (255.0-currentMaskGaussian)/255.0 #Creates weighted inversed mask

        LeftHalfBlendedLaplacian = currentFore*(weightsMaskGaussian)
        RightHalfBlendedLaplacian = currentBack*(invertedWeightsMaskGaussian)
        currentBlendedLaplacian = np.uint8(LeftHalfBlendedLaplacian+RightHalfBlendedLaplacian)

        BLENDEDLAPLACIAN = [currentBlendedLaplacian]+BLENDEDLAPLACIAN


    combinedimage = Reconstruct(np.array(BLENDEDLAPLACIAN), NLevels)
    cv2.imshow("Combined image:",combinedimage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


#Unused method
def cropAndCombineGivenImages(listofimagepaths:[str], NLevels:int=1,wantGrayImage:bool=False):
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
        #print(circles)
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


    combinedimage = croppedimages[0] #combined image starts with first cropped image
    print(combinedimage)
    for i in range(1,len(croppedimages)):
        if (croppedimages[i].shape[0] != combinedimage.shape[0] or croppedimages[i].shape[1] != combinedimage.shape[1]):
            croppedimages[i]= ScaleByGivenDimensions(croppedimages[i], (combinedimage.shape[1], combinedimage.shape[0]))

        combinedimage = cv2.addWeighted(combinedimage,0.5,croppedimages[i],0.5,0)


    cv2.imshow("Combined image:",combinedimage)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


#Allows the user to manually select images
def browseImagesDialog(startindirectory:str=os.getcwd(),MessageForUser='Please select a file'):
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    filepath = filedialog.askopenfilename(parent=root, initialdir=startindirectory, title=MessageForUser,filetypes = (("png files","*.png"),("jpeg files","*.jpg"),("all files","*.*")))
    return filepath


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

    #Sets up tkinter root windows and closed it (only need it for user browsing)
    root = tkinter.Tk()
    root.withdraw()

    #Allows user to manually pick their two images
    curr_directory = os.getcwd()
    image1path = browseImagesDialog(MessageForUser='Select your image1 (Foreground)')
    image2path = browseImagesDialog(MessageForUser='Select your image2 (Background)')


    #Gets a list of imagepaths (Strings) to all images contained within ./input_images folder:
    listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR,True)
    #LaplacianPyramids(getImageArray(listOfImages[4],False),4,True)

    #If wantGrayImage is false gets a colored image, otherwise gets a gray image
    wantGrayImage = False
    cropAndMaskGivenImages(image1path,image2path,wantGrayImage, NLevels=3)


main()
