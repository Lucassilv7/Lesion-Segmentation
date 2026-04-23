from matplotlib import pyplot as plt

from utils import load_image
from color_constancy import shades_of_gray, color_constancy
from digital_hair_removal import dhr, sharp_razor
from clahe import cl, clahe
from evaluation_metrics import calculate_metrics
import os
import numpy as np


if __name__ == "__main__":
    # Load the image
    img_bgr = load_image(r"C:\Users\lucas\UFERSA\Computer_Science\IC\Segmentation\Datasets\ISIC2018\Training\Input\ISIC_0000074.jpg")

    # Apply color constancy
    img_const = shades_of_gray(img_bgr)

    #color_constancy(img_bgr)

    # Aplique o clahe procurando um melhor valor para o clip limit, iterando entre 1.0 e 4.0 com passo de 0.5, logo depois passe pelo sharp_razor para remover os pelos. Salve cada resultado em uma pasta chamada "results_<clip_limit>" com o nome sendo o valor do clip limit utilizado.
    # Create output directory
    """  
    output_dir = "results_clip_limit"
    os.makedirs(output_dir, exist_ok=True)
    
    for clip_limit in np.arange(0.5, 2.5, 0.25):
        img_const_clahe = clahe(img_const, clip_limit=clip_limit)
        dhr_result = sharp_razor(img_const_clahe)

        # Save the result with clip limit value as filename
        output_path = os.path.join(output_dir, f"{clip_limit:.1f}.jpg")
        cv2.imwrite(output_path, dhr_result)
        print(f"Saved DHR result with clip limit {clip_limit:.1f} to {output_path}") 
        """
    
    """ output_dir = "results_dhr_clahe"
    os.makedirs(output_dir, exist_ok=True)

    img_const_dhr = sharp_razor(img_const)
    for clip_limit in np.arange(0.5, 2.5, 0.25):
        img_const_dhr_clahe = clahe(img_const_dhr, clip_limit=clip_limit)

        output_path = os.path.join(output_dir, f"{clip_limit:.1f}.jpg")
        cv2.imwrite(output_path, img_const_dhr_clahe)
        print(f"Saved CLAHE result with clip limit {clip_limit:.1f} to {output_path}") """


    """ img_const_clahe = clahe(img_const) """

    #dhr(img_const)
    
    #cl(img_const)


    img_const_dhr = sharp_razor(img_const)
    img_const_dhr_clahe = clahe(img_const_dhr, clip_limit=0.5)

    img_const_clahe = clahe(img_const, clip_limit=0.5)
    dhr_result = sharp_razor(img_const_clahe)
    dhr_result_clahe = clahe(dhr_result, clip_limit=0.5)


    # Display the results
    """ plt.figure(figsize=(20, 10))

    plt.subplot(2, 4, 1)
    plt.title("Sharp Razor result")
    plt.imshow(cv2.cvtColor(img_const_dhr, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.title("Sharp Razor + CLAHE")
    plt.imshow(cv2.cvtColor(img_const_dhr_clahe, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.title("CLAHE + Sharp Razor")
    plt.imshow(cv2.cvtColor(dhr_result, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.title("CLAHE + Sharp Razor + CLAHE")
    plt.imshow(cv2.cvtColor(dhr_result_clahe, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show() """

    # Compare the results using evaluation metrics
    # Using results_clip_limit folder.
    for i in np.arange(0.5, 2.5, 0.25):
        processed_image_path = os.path.join("results_clip_limit", f"{i:.1f}.jpg")
        processed_image = load_image(processed_image_path)
        ssim_val, psnr_val = calculate_metrics(img_bgr, processed_image)
        #Saved results in csv file with the name "evaluation_results_clip_limit.csv" with columns "clip_limit", "ssim", "psnr"
        with open("evaluation_results_clip_limit.csv", "a") as f:
            if i == 0.5:
                f.write("clip_limit,ssim,psnr\n")
            f.write(f"{i:.1f},{ssim_val:.4f},{psnr_val:.2f}\n")


    print("Evaluation to Digital Hair Removal completed.")

    # Using results_dhr_clahe folder.
    for i in np.arange(0.5, 2.5, 0.25):
        processed_image_path = os.path.join("results_dhr_clahe", f"{i:.1f}.jpg")
        processed_image = load_image(processed_image_path)
        ssim_val, psnr_val = calculate_metrics(img_bgr, processed_image)
        #Saved results in csv file with the name "evaluation_results_dhr_clahe.csv" with columns "clip_limit", "ssim", "psnr"
        with open("evaluation_results_dhr_clahe.csv", "a") as f:
            if i == 0.5:
                f.write("clip_limit,ssim,psnr\n")
            f.write(f"{i:.1f},{ssim_val:.4f},{psnr_val:.2f}\n")
    
    print("Evaluation to CLAHE + Digital Hair Removal completed.")
