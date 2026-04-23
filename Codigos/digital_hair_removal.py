import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_laplace
from skimage.filters import threshold_li
from skimage.measure import label, regionprops
import warnings  # Adicionado para tratar os avisos matemáticos

def dull_razor(img_bgr: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply closing with linear/cross structuring elements
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))  # Cross-shaped for multiple angles
    closed = cv2.morphologyEx(gray_bgr, cv2.MORPH_CLOSE, kernel)

    # Black-Hat
    blackhat = closed - gray_bgr

    # Threshold the blackhat image to create a mask of the hair
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the original image using the mask
    inpainted = cv2.inpaint(img_bgr, mask, 6, cv2.INPAINT_TELEA)

    # Apply adaptive median filter to smooth the final image
    smoothed = cv2.medianBlur(inpainted, 3)  # Adjust kernel size as needed

    return smoothed 

def canny_plus_coherence_transport(img_bgr: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Convert to float for Wiener filter
    gray_float = gray_bgr.astype(np.float64)

    # Filter the image to reduce noise (ignorando os avisos irritantes do console)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered_float = wiener(gray_float, (3, 3))

    # Remove NaN values that may arise from the Wiener filter
    filtered_float = np.nan_to_num(filtered_float, nan=0.0)
    
    # SEGURANÇA: Garante que os números não passem de 255 nem sejam menores que 0
    # e converte de volta para o formato de imagem (uint8)
    filtered_uint8 = np.clip(filtered_float, 0, 255).astype(np.uint8)

    # Calculate the median intensity of the image
    median = np.median(filtered_uint8)

    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))

    # Apply Canny detection
    edges = cv2.Canny(filtered_uint8, lower, upper)

    # Dilate the edges to create a thicker mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Inpaint the original image using the dilated edge mask    
    inpainted = cv2.inpaint(img_bgr, dilated_edges, 3, cv2.INPAINT_NS)

    return inpainted

def laplacian_of_gaussian(img_bgr: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian of Gaussian
    log = gaussian_laplace(gray_bgr, sigma=1)

    # Threshold the LoG result to create a binary mask
    _, mask = cv2.threshold(log.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)

    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

    # Inpaint the original image using the mask
    inpainted = cv2.inpaint(img_bgr, closed_mask, 3, cv2.INPAINT_TELEA)

    return inpainted

def sharp_razor(img_bgr: np.ndarray) -> tuple:
    # Focus on the red channel
    red_channel = img_bgr[:, :, 2]

    # Directional Gradients
    k_0 = np.array([[-1, -1, -1],
                    [ 2,  2,  2],
                    [-1, -1, -1]], dtype=np.float32)
                    
    k_45 = np.array([[-1, -1,  2],
                     [-1,  2, -1],
                     [ 2, -1, -1]], dtype=np.float32)
                     
    k_minus45 = np.array([[ 2, -1, -1],
                          [-1,  2, -1],
                          [-1, -1,  2]], dtype=np.float32)

    # Apply the filters to get directional responses
    response_0 = cv2.filter2D(red_channel, cv2.CV_32F, k_0)
    response_45 = cv2.filter2D(red_channel, cv2.CV_32F, k_45)
    response_minus45 = cv2.filter2D(red_channel, cv2.CV_32F, k_minus45)

    # Sum responses to get a combined edge strength
    summed_gradients = np.abs(response_0) + np.abs(response_45) + np.abs(response_minus45)

    # Normalize back to 0-255
    summed_gradients = cv2.normalize(summed_gradients, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Entropy Thresholding (Yen)
    threshold = threshold_li(summed_gradients)

    # Create a binary mask of hair regions
    brute_mask = summed_gradients > threshold
    brute_mask = brute_mask.astype(np.uint8) * 255

    # Geometric Filtering
    labeled_mask = label(brute_mask)
    props = regionprops(labeled_mask)

    # Create zero mask
    clean_mask = np.zeros_like(brute_mask)

    # Avaliate object
    for prop in props:
        if prop.area >= 10 and prop.solidity <= 0.9:
            for r, c in prop.coords:
                clean_mask[r, c] = 255

    # Dilate the clean mask to cover the hair regions more completely
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.dilate(clean_mask, kernel_dilate, iterations=1)

    # Inpaint the original image using the clean mask
    inpainted = cv2.inpaint(img_bgr, clean_mask, 3, cv2.INPAINT_NS)

    return inpainted

def dhr(img_bgr: np.ndarray):
  
    dull_razor_result = dull_razor(img_bgr)

    canny_result = canny_plus_coherence_transport(img_bgr)

    log_result = laplacian_of_gaussian(img_bgr)

    sharp_razor_result = sharp_razor(img_bgr)

    print("Processing complete. Displaying results...")
    # Display the original and processed images
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Dull Razor Result")
    plt.imshow(cv2.cvtColor(dull_razor_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Canny + Coherence Transport Result")
    plt.imshow(cv2.cvtColor(canny_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Laplacian of Gaussian Result")
    plt.imshow(cv2.cvtColor(log_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Sharp Razor Result")
    plt.imshow(cv2.cvtColor(sharp_razor_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Show Image Original and Sharp Razor Result (Inpainted, Brute Mask and Clean Mask)
    """ plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Sharp Razor Result")
    plt.imshow(cv2.cvtColor(sharp_razor_result, cv2.COLOR_BGR2RGB)) 
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Brute Mask")
    plt.imshow(brute_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Clean Mask")
    plt.imshow(clean_mask, cmap='gray') 
    plt.axis('off')

    plt.tight_layout()  
    plt.show() """