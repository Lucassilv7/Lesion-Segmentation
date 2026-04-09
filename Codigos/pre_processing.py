import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray_world(img_bgr):
    
    img_b , img_g , img_r = cv2.split(img_bgr)   
    
    # Calculate the average values for each channel
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

def dullrazor(img_bgr):
    # Convert to grayscale
    grayScale = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Use morphological operations to find the hair-like structures
    kernel = cv2.getStructuringElement(1, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    
    # Threshold the blackhat image to create a mask of the hair
    _, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the original image using the mask
    inpainted = cv2.inpaint(img_bgr, mask, 6, cv2.INPAINT_TELEA)
    
    return inpainted

def clahe(img_bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    # Convert to LAB color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l_channel)

    # Merge the channels back
    lab = cv2.merge([cl, a, b])
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img_clahe

if __name__ == "__main__":
    # Load the image
    img_bgr = cv2.imread(r"C:\Users\lucas\UFERSA\Computer_Science\IC\Segmentation\Datasets\ISIC2018\Training\Input\ISIC_0000074.jpg")
    
    if img_bgr is None:
        print("Error: Could not load image. Please check the file path and format.")
        exit(1)
    
    # Apply gray world algorithm
    img_corrected_gray = gray_world(img_bgr)

    # Apply dull razor algorithm
    img_corrected_dull = dullrazor(img_corrected_gray)

    # Apply CLAHE
    img_corrected_clahe = clahe(img_corrected_dull)
    
    # Display the original and corrected images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Gray World Corrected Image')
    plt.imshow(cv2.cvtColor(img_corrected_gray, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Dull Razor Corrected Image')
    plt.imshow(cv2.cvtColor(img_corrected_dull, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('CLAHE Corrected Image')
    plt.imshow(cv2.cvtColor(img_corrected_clahe, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()