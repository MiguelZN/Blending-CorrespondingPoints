# PART2---------------------------------------------------------------------------------
import numpy as np
import cv2
import os
from cpselect.cpselect import cpselect

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


# Finds the image 2 point given an image 1 point and a set of known points corresponding from image 1 to image 2
def findCorrespondingImage2Point(point, image1image2points: dict):
    # The image one points matrix: (used to find affine parameters)
    imageOnePoints_matrix = np.zeros((2 * len(image1image2points), 6))

    # The image two points for image one points: (used to find affine parameters)
    resultsTwoPoints_matrix = np.zeros((2 * len(image1image2points), 1))

    # Creates the 2d arrays for n amount of points
    for i in range(len(image1image2points)):
        imageOnePoints_matrix[i * 2] = [image1image2points[i]['img1_x'], image1image2points[i]['img1_y'], 0, 0, 1, 0]
        imageOnePoints_matrix[(i * 2) + 1] = [0, 0, image1image2points[i]['img1_x'], image1image2points[i]['img1_y'], 0,
                                              1]

        resultsTwoPoints_matrix[i * 2] = image1image2points[i]['img2_x']
        resultsTwoPoints_matrix[(i * 2) + 1] = image1image2points[i]['img2_y']

    m1m2m3m4txty_matrix = None

    # If the image1image2points contains more than 3 points then fixes overconstraint:
    if (3 < len(image1image2points)):
        m1m2m3m4txty_matrix = (np.linalg.inv(
            imageOnePoints_matrix.T @ imageOnePoints_matrix) @ imageOnePoints_matrix.T) @ resultsTwoPoints_matrix

    # If the image1image2points contain 3 points then multiply by the inverse matrix
    elif (len(image1image2points) == 3):
        m1m2m3m4txty_matrix = np.linalg.inv(imageOnePoints_matrix) @ resultsTwoPoints_matrix
    else:
        raise Exception("ERROR:The image1image2points are neither 3 points or more!")

    print('Affine parameters of points from source image to target image are:')
    print(m1m2m3m4txty_matrix)

    # Pulls out the transformation parameters:
    m1m2m3m4_matrix = np.array([
        [m1m2m3m4txty_matrix[0][0], m1m2m3m4txty_matrix[1][0]],
        [m1m2m3m4txty_matrix[2][0], m1m2m3m4txty_matrix[3][0]]])
    txty_matrix = np.array([
        [m1m2m3m4txty_matrix[4][0]],
        [m1m2m3m4txty_matrix[5][0]]
    ])

    # print("m1m2m3m4_matrix:")
    # print(m1m2m3m4_matrix)
    # print(txty_matrix.shape)

    # Converts image1 point into matrix equivalent:
    image1Point = np.array([
        [point[0]],
        [point[1]]
    ])

    # print("imageOnePoints_matrix:")
    # print(imageOnePoints_matrix)
    # print(image1Point)

    # Calculates the corresponding image 2 point by the transformation paramters and image 1 point:
    correspondingImage2Point = (m1m2m3m4_matrix @ image1Point) + txty_matrix

    print("Old Point:[" + str(image1Point[0][0]) + "," + str(image1Point[1][0]) + "]")
    print("New Point:[" + str(correspondingImage2Point[0][0]) + "," + str(correspondingImage2Point[1][0]) + "]")

    return correspondingImage2Point


# Gets corresponding images 1,2:
listOfImages = getAllImagesFromInputImagesDir(IMAGEDIR,True)
mountainimage1 = getImageFromListOfImages(listOfImages, 'im1_1')
mountainimage2 = getImageFromListOfImages(listOfImages, 'im1_2')

# (These two set of points already contain points selected from cpselect)
threeimage1image2points = [
    {'point_id': 1, 'img1_x': 209.89760003380235, 'img1_y': 81.64180081970682, 'img2_x': 64.54077407360467,
     'img2_y': 99.30352389402975},
    {'point_id': 2, 'img1_x': 304.71316601174635, 'img1_y': 169.95041619132132, 'img2_x': 154.70851818988467,
     'img2_y': 193.18952549964092},
    {'point_id': 3, 'img1_x': 246.15061055478094, 'img1_y': 210.8512485739638, 'img2_x': 93.35726961592093,
     'img2_y': 233.16079350995062}]
morethanthreeimage1image2points = [
    {'point_id': 1, 'img1_x': 209.89760003380235, 'img1_y': 81.64180081970682, 'img2_x': 64.54077407360467,
     'img2_y': 99.30352389402975},
    {'point_id': 2, 'img1_x': 304.71316601174635, 'img1_y': 169.95041619132132, 'img2_x': 154.70851818988467,
     'img2_y': 193.18952549964092},
    {'point_id': 3, 'img1_x': 246.15061055478094, 'img1_y': 210.8512485739638, 'img2_x': 93.35726961592093,
     'img2_y': 233.16079350995062},
    {'point_id': 4, 'img1_x': 201.5315206828073, 'img1_y': 150.42956437233283, 'img2_x': 51.52687286094567,
     'img2_y': 170.8799805636541},
    {'point_id': 5, 'img1_x': 393.0217813833608, 'img1_y': 47.247919043393836, 'img2_x': 243.01713356149924,
     'img2_y': 78.85310770270848}]
inverse_morethanthreeimage1image2points = [
    {'point_id': 1, 'img2_x': 209.89760003380235, 'img2_y': 81.64180081970682, 'img1_x': 64.54077407360467,
     'img1_y': 99.30352389402975},
    {'point_id': 2, 'img2_x': 304.71316601174635, 'img2_y': 169.95041619132132, 'img1_x': 154.70851818988467,
     'img1_y': 193.18952549964092},
    {'point_id': 3, 'img2_x': 246.15061055478094, 'img2_y': 210.8512485739638, 'img1_x': 93.35726961592093,
     'img1_y': 233.16079350995062},
    {'point_id': 4, 'img2_x': 201.5315206828073, 'img2_y': 150.42956437233283, 'img1_x': 51.52687286094567,
     'img1_y': 170.8799805636541},
    {'point_id': 5, 'img2_x': 393.0217813833608, 'img2_y': 47.247919043393836, 'img1_x': 243.01713356149924,
     'img1_y': 78.85310770270848}]

selectedPoints = morethanthreeimage1image2points
# print(selectedPoints)

# (Uncomment below to get points manually using cpselect)
selectedPoints = cpselect(mountainimage1,mountainimage2)

# Test point (verified to be about the same feature in image 1,2)
point4_image1 = [morethanthreeimage1image2points[4]['img1_x'], morethanthreeimage1image2points[4]['img1_y']]
point4_image2 = [morethanthreeimage1image2points[4]['img2_x'], morethanthreeimage1image2points[4]['img2_y']]

findCorrespondingImage2Point(point4_image1, selectedPoints)
print("Selected:" + str(point4_image2) + '\n')

findCorrespondingImage2Point(point4_image1, morethanthreeimage1image2points)
print("Actual:" + str(point4_image2) + '\n')

print("Inverse:")
findCorrespondingImage2Point(point4_image2, inverse_morethanthreeimage1image2points)
print("Actual:" + str(point4_image1) + '\n')

findCorrespondingImage2Point((366.06,107.67), selectedPoints)
print("Actual:216.06, 135.56"+ '\n')
# #cpselect(mountainimage1, mountainimage2)
# #366.06,107.67  216.06, 135.56
#
# findCorrespondingImage2Point((361.42, 133.7), selectedPoints)
# print("Actual:211.41, 159.73" + '\n')
# 361.42, 133.7, 211.41, 159.73

# Displaying Click Points on image1:
image1_forpoints = getImageArray(mountainimage1, False)
image2_forpoints = getImageArray(mountainimage2, False)

for pointDict in morethanthreeimage1image2points:
    cv2.circle(image1_forpoints, (int(pointDict['img1_x']), int(pointDict['img1_y'])), 2, (0, 0, 255), -1)
    cv2.namedWindow("image1")
    # cv2.setMouseCallback("image1", mouse_drawing)
print("Displaying image 1 points for two seconds..")
displayImageGivenArray(image1_forpoints, waitKey=2000)

for pointDict in morethanthreeimage1image2points:
    cv2.circle(image2_forpoints, (int(pointDict['img2_x']), int(pointDict['img2_y'])), 2, (0, 0, 255), -1)
    cv2.namedWindow("image1")
    # cv2.setMouseCallback("image1", mouse_drawing)
print("Displaying image 2 points for two seconds..")
displayImageGivenArray(image2_forpoints, waitKey=2000)