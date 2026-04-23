import cv2
import numpy as np
import matplotlib.pyplot as plt

def clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
    # Convert to LAB color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create a CLAHE object and apply it to the L channel
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_l_channel = clahe_obj.apply(l_channel)

    # Merge the CLAHE enhanced L channel back with the original A and B channels
    merged_lab = cv2.merge((cl_l_channel, a_channel, b_channel))

    # Convert back to BGR color space
    enhanced_img_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return enhanced_img_bgr

def cl(img_bgr: np.ndarray):
    
    cl = clahe(img_bgr)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("CLAHE Enhanced Image")
    plt.imshow(cv2.cvtColor(cl, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()