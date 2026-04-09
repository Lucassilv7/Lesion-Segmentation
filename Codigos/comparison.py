from utils import load_image
from color_constancy import shades_of_gray, color_constancy
from digital_hair_removal import dhr
import cv2

if __name__ == "__main__":
    # Load the image
    img_bgr = load_image(r"C:\Users\lucas\UFERSA\Computer_Science\IC\Segmentation\Datasets\ISIC2018\Training\Input\ISIC_0000074.jpg")

    # Apply color constancy
    img_const = shades_of_gray(img_bgr)

    #color_constancy(img_bgr)

    dhr(img_const)
    