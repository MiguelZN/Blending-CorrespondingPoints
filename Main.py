import numpy as np
import cv2
import os
import subprocess
IMAGEDIR = './input_images/'

import subprocess



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






def displayImage(imagepath:str):
    img = getImageArray(imagepath)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getImageArray(imagepath:str):
    return cv2.imread(imagepath)

#Part 1:------------------------------
'''
I is image of varying size, 
H is kernel of varying size. 
Output: convolution result that is displayed. 
'''
def Convolve(I,H):
    return

'''
takes image I as input and outputs copy of image re-sampled
by half the width and height of the input. Remember to 
Gaussian filter the image before reducing it; use separable
1D Gaussian kernels. 
'''
def Reduce(I):
    return

'''
takes image I as input and outputs copy of image expanded,
twice the width and height of the input. '''
def Expand(I):
    return


'''
takes in an image I,
takes in N (int) is the no. of levels. 
'''
def GaussianPyrmaid(I,N):
    return

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

    displayImage(listOfImages[0])
    displayImage(listOfImages[1])
    displayImage(listOfImages[2])
main()