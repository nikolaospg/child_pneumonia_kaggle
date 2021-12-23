import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


#Takes as input the path to a folder (corresponding to one of the four Folders where the datasets are saved), and gathers the heights and the widths
#of each image of the specific dataset. Creates the histograms and a scatter plot, and finally returns the mean height and the mean width
def make_histograms(path_to_folder, title):

    os.chdir(path_to_folder)            #Getting to the folder specified as a parameter
    num_images=len(os.listdir())        #The total number of images in the directory
    height_array=np.zeros(shape=(num_images,))
    width_array=np.zeros(shape=(num_images,))


    #For every image of the folder:
    count=0
    for file in os.listdir():
        #Reading the image
        current_image=cv.imread(file, 0)     

                 
        current_dims=current_image.shape
        
        height_array[count]=current_dims[0]
        width_array[count]=current_dims[1]
        count=count+1
    mean_height=np.mean(height_array)
    mean_width=np.mean(width_array)

    plt.figure(1)
    plt.title(title)
    plt.scatter(height_array, width_array, marker=".")
    plt.xlabel("Height")
    plt.ylabel("Width")
    plt.show()

    plt.figure(2)
    plt.title(title+" Height")
    plt.hist(height_array, bins=50)
    plt.show()

    plt.figure(3)
    plt.title(title+" Width")
    plt.hist(width_array, bins=50)
    plt.show()


    return mean_height,mean_width,num_images



#Uncomment one of the following:

# path_to_folder="Pediatric_Chest_X-ray_Pneumonia/train/NORMAL"           #The path on my system   
# mean_height,mean_width,num_images=make_histograms(path_to_folder, "Normal, Train set, ")
# print("Mean height is %f, mean width is %f, ratio is %f and num_images is %d - Train Normal" % (mean_height, mean_width,mean_height/mean_width, num_images))
 
path_to_folder="Pediatric_Chest_X-ray_Pneumonia/train/PNEUMONIA"     
mean_height,mean_width,num_images=make_histograms(path_to_folder, "Pneumonia, Train set, ")
print("Mean height is %f, mean width is %f, ratio is %f and num_images is %d - Train Pneumonia" % (mean_height, mean_width,mean_height/mean_width, num_images))

# path_to_folder="Pediatric_Chest_X-ray_Pneumonia/test/NORMAL"    
# mean_height,mean_width,num_images=make_histograms(path_to_folder, "Normal, Test set, ")
# print("Mean height is %f, mean width is %f, ratio is %f and num_images is %d - Test Normal" % (mean_height, mean_width,mean_height/mean_width, num_images))

# path_to_folder="Pediatric_Chest_X-ray_Pneumonia/test/PNEUMONIA"    
# mean_height,mean_width,num_images=make_histograms(path_to_folder, "Pneumonia, Test set, ")
# print("Mean height is %f, mean width is %f, ratio is %f and num_images is %d - Test Pneumonia" % (mean_height, mean_width,mean_height/mean_width, num_images))
