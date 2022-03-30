import os
import numpy as np
import cv2 as cv
import time

#This script is used to get the descriptors from the images. They are saved, and we can later use visual_vocabulary.py to get the features

#This function takes an image as an input and computes the SIFT keypoints+descriptors. 
#The preprocessing part takes places here (according to the flags passed by the user)/
#We use the opencv implementation for the SIFT algorithm, and we use can histogram equalisation and gaussian blurring if we want (judging on the keypoints that are derived)
#show_flag-> shows the image with the keypoints extracted
def image_extract(image, show_flag, hist_flag, blur_flag, descriptors_per_image, filter_dims=(5,5)):

    if(hist_flag==1):
        image=cv.equalizeHist(image) 

    if(blur_flag==1):
        image=cv.GaussianBlur(image,filter_dims,cv.BORDER_DEFAULT)
    
    sift = cv.SIFT_create()
    points, features= sift.detectAndCompute(image,None)
    current_num_descriptors=len(features)
    if(show_flag==1):
        print("We have %d keypoints" % (current_num_descriptors))
        image1=cv.drawKeypoints(image,points,image, color=[0,255,0])
        cv.imshow("",image1)
        cv.waitKey(0)
    
    #Undersampling the features
    #In case the number of descriptors is less than the descriptors per image we asked for
    if(current_num_descriptors<descriptors_per_image):
        features_ret=np.empty(shape=[descriptors_per_image,128])
        features_ret[:]=np.nan
        features_ret[0:current_num_descriptors]=features

    else:
        perm=np.random.permutation(current_num_descriptors)
        perm=perm[0:descriptors_per_image]
        features_ret=features[perm,:]
    return features_ret

#Function Used to read the images and get the keypoint descriptors, for the images on one specific folder
#Final_images-> Parameter used to control the amount of images we are going to use on this dataset
def folder_descriptors(path_to_folder, descriptors_per_image, final_images=None):

    os.chdir(path_to_folder)            #Getting to the folder specified as a parameter

    #If the argument is none then we will not be reducing the dataset
    if(final_images==None):
        num_images=len(os.listdir())        #The total number of images in the directory
    else:
        num_images=final_images
    
    current_descriptors=np.zeros((num_images,descriptors_per_image,128), dtype="float32")
    
    count=0
    #For every image of the folder:
    for file in os.listdir():
        if(count==num_images):  #This if clause is used to apply the dataset reduction
            break

        current_image=cv.imread(file, 0)   
        features=image_extract(current_image, show_flag=0, hist_flag=1, blur_flag=1, descriptors_per_image=descriptors_per_image)       #Extracting features from the image
        current_descriptors[count]=features
        count=count+1
    return current_descriptors


#Getting the descriptors:
descriptors_per_image=2000
init_directory=os.getcwd()

#In case we want to tackle class imbalance, we get rid of image from the pneumonia train set
#Uncomment one of the following choices(first one is the reduced dataset, the second is the original)
final_pn_train=1750          #Max is 3883

#First for the train. I do the train and the test separately mainly for memory reasons
t1=time.time()
path_to_folder="Pediatric_Chest_X-ray_Pneumonia/train/PNEUMONIA"           #The path on my system   
pneumonia_descriptors=folder_descriptors(path_to_folder, descriptors_per_image, final_pn_train)
os.chdir(init_directory) #Returning to the initial folder

path_to_folder="Pediatric_Chest_X-ray_Pneumonia/train/NORMAL"           #The path on my system   
normal_descriptors=folder_descriptors(path_to_folder, descriptors_per_image)
os.chdir(init_directory)

train_descriptors=np.concatenate((normal_descriptors, pneumonia_descriptors), axis=0)       #Merging the two


# #Now for the test:
# path_to_folder="Pediatric_Chest_X-ray_Pneumonia/test/PNEUMONIA"           #The path on my system   
# pneumonia_descriptors=folder_descriptors(path_to_folder, descriptors_per_image)
# os.chdir(init_directory) #Returning to the initial folder

# path_to_folder="Pediatric_Chest_X-ray_Pneumonia/test/NORMAL"           #The path on my system   
# normal_descriptors=folder_descriptors(path_to_folder, descriptors_per_image)
# os.chdir(init_directory)
# t2=time.time()


# test_descriptors=np.concatenate((normal_descriptors, pneumonia_descriptors), axis=0)       #Merging the two
# print("Time needed for extraction (+concatenation) is",t2-t1)



##Saving the descriptors
np.save("descriptors/REDdescriptors_train"+str(descriptors_per_image), train_descriptors)
# np.save("descriptors/descriptors_test"+str(descriptors_per_image), test_descriptors)

