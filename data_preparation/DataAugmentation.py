
import cv2
import random
import os
from scipy.ndimage.filters import median_filter


class Data_augmentation:
    def __init__(self, path, img):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        self.path = path
        self.image = img  # cv2.imread(path+image_name)

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        if angle < 0 and False:
            angle = 360 + angle

        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def crop_image(self, image, y1=0, y2=254, x1=0, x2=254):
        # print(image.shape)
        image = image[y1:y2, x1:x2]
        # print(image.shape)
        image = cv2.resize(image, (255, 255))
        # print(image.shape)
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def CLAHE(self, img, gridsize=8):

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(gridsize, gridsize))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def transform_img_and_denoise(self, img, img_width=255, img_height=255):

        # Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        # Image Denoising
        #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

        # Image Resizing
        #height, width, channels = image.shape
        #img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

        return img

    def sharpen_image(self, img, sigma, strength):
        # Image Median filtering
        img_mf = median_filter(img, sigma)

        # Laplacial Calculation
        lap = cv2.Laplacian(img_mf, cv2.CV_64F)

        # Sharpen the Image
        sharp = img-strength*lap

        # Saturate Pixels

        sharp[sharp > 255] = 255
        sharp[sharp < 0] = 0

        return sharp

    def image_augment(self, save_path, i):
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        img1 = self.image.copy()
        #img = self.sharpen_image(self.transform_img_and_denoise(img1),sigma=5,strength=0.8)

        img = self.sharpen_image(
            self.CLAHE(img1,8),
            sigma=5,
            strength=0.8
        )
        #img_crop = self.rotate(img, 0, 1.1)
        #img_rot = self.rotate(img, 4, 1.2)
        #img_rot2 = self.rotate(img, -4, +1.2)

        cv2.imwrite(os.path.join(save_path, str(i)+'.jpg'), img)
        #cv2.imwrite(os.path.join(save_path, str(i + 1)+'.jpg'), img_crop)
        #cv2.imwrite(os.path.join(save_path, str(i + 2)+'.jpg'), img_rot)
        #cv2.imwrite(os.path.join(save_path, str(i + 3)+'.jpg'), img_rot2)


def img_aug(root, img, output_path, i):
    raw_image = Data_augmentation(root, img)
    raw_image.image_augment(output_path, i)
    return 1
