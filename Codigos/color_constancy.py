import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_world(img_bgr):
    
    img_b , img_g , img_r = cv2.split(img_bgr)   
    
    # Calculate the average values for each channo q el
    avg_r = np.mean(img_r)
    avg_g = np.mean(img_g)
    avg_b = np.mean(img_b)
    
    # Calculate the overall average
    avg_all = (avg_r + avg_g + avg_b) / 3
    
    # Calculate the scaling factors for each channel
    scale_r = np.clip(img_r * (avg_all / avg_r), 0, 255).astype(np.uint8)
    scale_b = np.clip(img_b * (avg_all / avg_b), 0, 255).astype(np.uint8)
    scale_g = np.clip(img_g * (avg_all / avg_g), 0, 255).astype(np.uint8)
    
    return cv2.merge([scale_b, scale_g, scale_r])

def shades_of_gray(img_bgr: np.ndarray, p: int = 6):
    
    img_b, img_g, img_r = cv2.split(img_bgr)

    # Apply the power function to each channel
    img_b_elevated = np.power(img_b.astype(np.float32), p)
    img_g_elevated = np.power(img_g.astype(np.float32), p)
    img_r_elevated = np.power(img_r.astype(np.float32), p)

    # Calculate the average values for each channel
    avg_r = np.mean(img_r_elevated)
    avg_g = np.mean(img_g_elevated)
    avg_b = np.mean(img_b_elevated)

    # Calculate the p-th root of the average values
    root_r = np.power(avg_r, 1/p)
    root_g = np.power(avg_g, 1/p)
    root_b = np.power(avg_b, 1/p)

    # Get maximum value among the channels
    max_ilum = max(root_r, root_g, root_b)

    # Scale the original channels by the ratio of the maximum value to the p-th root of the average values
    ilum_r = img_r.astype(np.float32) * (max_ilum / root_r)
    ilum_g = img_g.astype(np.float32) * (max_ilum / root_g)
    ilum_b = img_b.astype(np.float32) * (max_ilum / root_b)

    # Clip the values to the valid range [0, 255] and convert back to uint8
    ilum_r = np.clip(ilum_r, 0, 255).astype(np.uint8)
    ilum_g = np.clip(ilum_g, 0, 255).astype(np.uint8)
    ilum_b = np.clip(ilum_b, 0, 255).astype(np.uint8)

    # Merge the channels back into a BGR image
    return cv2.merge([ilum_b, ilum_g, ilum_r])

def color_constancy(img_bgr: np.ndarray):

    img_gray_shades = shades_of_gray(img_bgr, p=1)

    img_shades = shades_of_gray(img_bgr)

    img_gray_world = gray_world(img_bgr)

    # Display the original and corrected images
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title('Gray World by shades\nof gray algorithm (p=1)')
    plt.imshow(cv2.cvtColor(img_gray_shades, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Shades of Gray\nCorrected Image')
    plt.imshow(cv2.cvtColor(img_shades, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Gray World by\ngray world algorithm')
    plt.imshow(cv2.cvtColor(img_gray_world, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


