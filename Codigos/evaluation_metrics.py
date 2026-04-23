from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

def calculate_metrics(original: np.ndarray, processed: np.ndarray) -> float:
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim_value = ssim(original_gray, processed_gray)

    # Calculate PSNR
    psnr_value = psnr(original_gray, processed_gray)

    return ssim_value, psnr_value

