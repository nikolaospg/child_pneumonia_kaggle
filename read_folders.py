import os 
import numpy as np
import cv2 as cv

#This function takes as input the path to one of the 4 directories in which the images are stored and 
#returns a numpy array where every row corresponds to one image. Also the class label is stored in the last element of this row 
#Inputs:
#   1-> path_to folder: The path to the folder
#   2-> pneumonia_flag: This tells us whether this folder has the patterns for the "normal" class or the "pneumonia" class. 1 corresponds to pneumonia and 0 to normal
#   3-> image_dims: The dimensions of the numpy array
#   4-> path_to_log (optional) : This creates a log file which tells us the order by which the images are read. Useful for testing (on whether the conversion is correct)
#   5-> log_header (optional) : This is the header for the log file
def read_folder(path_to_folder, pneumonia_flag, image_dims, path_to_log=None, log_header=None):

    os.chdir(path_to_folder)            #Getting to the folder specified as a parameter
    vectorised_image_length=image_dims[0]*image_dims[1]         #The size of the vectors to be put in the final dataset (in each row)
    num_images=len(os.listdir())        #The total number of images in the directory
    final_dataset=np.zeros( [num_images, vectorised_image_length +1], dtype="uint8")      #Initialising the final dataset

    if((pneumonia_flag!=0) &  (pneumonia_flag!=1)):
        raise NameError('pneumonia_flag should be 0 or 1! Exiting ')

    #If the user specified a path_to_log, then create the file 
    if(path_to_log!=None):
        os.system("touch %s" % path_to_log)
        os.system("echo %s >> %s" % (log_header, path_to_log))


    #For every image of the folder:
    count=0
    for file in os.listdir():
        
        #Putting the name of the current file on the log file
        if(path_to_log!=None):
            os.system("echo '%d -> %s' >> %s" % (count, file, path_to_log))
        
        #Reading the image
        current_image=cv.imread(file, 0)              


        #Resizing the image
        resize_dims=[ image_dims[1], image_dims[0] ]
        current_image=cv.resize(current_image, resize_dims, interpolation=cv.INTER_AREA)

        #Reshaping to a vector
        current_image=np.reshape(current_image, (vectorised_image_length,))


        #Adding class info and informing the final_dataset
        current_image=np.append(current_image, pneumonia_flag)
        final_dataset[count]=current_image

        count=count+1
    return final_dataset



#First getting the train set#
image_dims=np.array([30,40])

#Normal
path_to_folder="Pediatric_Chest_X-ray_Pneumonia/train/NORMAL"     #The path in my system

pneumonia_flag=0
pneumonia_flag=np.array(pneumonia_flag,dtype="uint8")           #Converting to uint8


path_to_log="~/Documents/TrainNormal"                                                                   #The path in my system
log_header="Showing the images converted on Train Normal"

normal_train_30x40=read_folder(path_to_folder, pneumonia_flag, image_dims, path_to_log, log_header)
os.chdir("../../..")
#Finished with normal Train

#Pneumonia
path_to_folder="Pediatric_Chest_X-ray_Pneumonia/train/PNEUMONIA"     #The path in my system

pneumonia_flag=1
pneumonia_flag=np.array(pneumonia_flag,dtype="uint8")           #Converting to uint8


path_to_log="~/Documents/TrainPneumonia"                                                                   #The path in my system
log_header="Showing the images converted on Train Pneumonia"

pneumonia_train_30x40=read_folder(path_to_folder, pneumonia_flag, image_dims, path_to_log, log_header)
os.chdir("../../..")
#Finished with pneumonia train

#Merging the Train set
train_set=np.concatenate((normal_train_30x40, pneumonia_train_30x40), axis=0)
np.save("train_set_30x40", train_set)



#Now working with the test set
path_to_folder="Pediatric_Chest_X-ray_Pneumonia/test/NORMAL"     #The path in my system

pneumonia_flag=0
pneumonia_flag=np.array(pneumonia_flag,dtype="uint8")           #Converting to uint8


path_to_log="~/Documents/TestNormal"                                                                   #The path in my system
log_header="Showing the images converted on Test Normal"

normal_test_30x40=read_folder(path_to_folder, pneumonia_flag, image_dims, path_to_log, log_header)
os.chdir("../../..")
#Finished with normal Test

#Pneumonia
path_to_folder="Pediatric_Chest_X-ray_Pneumonia/test/PNEUMONIA"     #The path in my system

pneumonia_flag=1
pneumonia_flag=np.array(pneumonia_flag,dtype="uint8")           #Converting to uint8


path_to_log="~/Documents/TestPneumonia"                                                                   #The path in my system
log_header="Showing the images converted on Test Pneumonia"

pneumonia_test_30x40=read_folder(path_to_folder, pneumonia_flag, image_dims, path_to_log, log_header)
os.chdir("../../..")
#Finished with pneumonia Test

#Merging the Test set
test_set=np.concatenate((normal_test_30x40, pneumonia_test_30x40), axis=0)
print(test_set.shape)
print(test_set.dtype)
np.save("test_set_30x40", test_set)