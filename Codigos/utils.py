import cv2
import numpy as np

def load_image(path: str) -> np.darray:
    # Load the image
    img_bgr = cv2.imread(path)
    
    if img_bgr is None:
        print("Error: Could not load image. Please check the file path and format.")
        exit(1)
    
    return img_bgr