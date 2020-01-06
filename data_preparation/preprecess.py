
import os
import cv2

from scipy.ndimage.filters import median_filter

#Size of images
IMAGE_WIDTH = 1920#395#1920#227
IMAGE_HEIGHT = 960#264#960#227

def transform_img_and_denoise(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])


    #Image Denoising
    #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    #Image Resizing
    #img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def sharpen_image(img, sigma, strength):
    #Image Median filtering 
    img_mf = median_filter(img, sigma)

    #Laplacial Calculation
    lap = cv2.Laplacian(img_mf, cv2.CV_64F)

    #Sharpen the Image
    sharp = img-strength*lap

    #Saturate Pixels

    sharp[sharp>255] =255
    sharp[sharp<0] = 0

    return sharp


#parent_dir = "/home/pakoxtror/Desktop/"
#directory = "pruebaeq"
#path = os.path.join(parent_dir, directory)
#os.mkdir(path)

""" for file in os.listdir("/home/pakoxtror/Desktop/prueba"):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        direc = "/home/pakoxtror/Desktop/prueba/"+ filename
        print(direc)
        img = cv2.imread(direc, cv2.IMREAD_COLOR)
        img = transform_img_and_denoise(img, IMAGE_WIDTH, IMAGE_HEIGHT)
        img = sharpen_image(img, 5, 0.8)
        saveas = "/home/pakoxtror/Desktop/pruebaeq/" + filename
        cv2.imwrite(saveas , img) """
img = cv2.imread("elena.jpg", cv2.IMREAD_COLOR)
height, width, channels = img.shape 
print(img.shape)
img1 = transform_img_and_denoise(img, IMAGE_WIDTH, IMAGE_HEIGHT)
cv2.imwrite("elena_1.jpg" , img1)
img2 = sharpen_image(img1, 5, 0.8)
cv2.imwrite("elena_2.jpg" , img2)
