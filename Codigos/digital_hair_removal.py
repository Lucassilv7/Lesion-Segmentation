import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

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
    smoothed = cv2.medianBlur(inpainted, 7)  # Adjust kernel size as needed

    return smoothed 

def canny_plus_coherence_transport(img_bgr: np.ndarray) -> np.ndarray:

    # Convert to grayscale
    gray_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Filter the image to reduce noise 
    filtered = cv2.GaussianBlur(gray_bgr, (3, 3), 0)
    
    # Calculate the median intensity of the image
    median = np.median(filtered)

    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))

    # Apply Canny detection
    edges = cv2.Canny(filtered, lower, upper)

    # Dilate the edges to create a thicker mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    dilated_edges = cv2.dilate(opened_edges, kernel, iterations=1)

    # Inpaint the original image using the dilated edge mask    
    inpainted = cv2.inpaint(img_bgr, dilated_edges, 3, cv2.INPAINT_NS)

    return inpainted

def dhr(img_bgr: np.ndarray):

    # Apply Dull Razor
    dull_razor_result = dull_razor(img_bgr)

    # Apply Canny + Coherence Transport
    canny_result = canny_plus_coherence_transport(img_bgr)

    # Display the original and processed images
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Dull Razor Result")
    plt.imshow(cv2.cvtColor(dull_razor_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Canny + Coherence Transport Result")
    plt.imshow(cv2.cvtColor(canny_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()