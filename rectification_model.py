# @Time    : 7/3/2023 11:03
# @Author  : wyq2985
# @File    : main.py
# @Software: PyCharm

import numpy as np
import cv2
from noiseEstimation import patch_noise_cal, patch_weight_cal, random_patch_selection, img_noise_cal


def noise_injection(image, sigma):
    """

    :param image_path: the target image
    :param sigma: the known sigma added to the image
    :return: a new noisy image for estimation
    """
    # Generate Gaussian noise with the same shape as the image
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)

    # Add the noise to the image
    noisy_image = np.clip(image + noise, 0, 1)

    # Convert the image back to the range of 0-255 and to the unsigned 8-bit integer type
    noisy_image = (noisy_image * 255).astype(np.uint8)

    # # Convert the image from BGR to RGB
    noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    # filename = f"injected_noise{sigma}.png"
    # cv2.imwrite(filename, noisy_image_rgb)

    return noisy_image_rgb


def new_estimation(new_img, overexposure_threshold, underexposure_threshold):
    blurred_image = cv2.GaussianBlur(new_img, (0, 0), 5.0)

    new_patches, blur_patches = random_patch_selection(new_img, blurred_image, 1000,
                                                               overexposure_threshold,
                                                               underexposure_threshold, patch_size=5)
    new_patch_noise = patch_noise_cal(new_patches)

    new_patch_weight = patch_weight_cal(blur_patches, gamma=2.0)

    noise_level = img_noise_cal(new_patch_noise, new_patch_weight)  # it's the variance

    return noise_level


def rectification_linear_model(noise_1, noise_2, inject_noise):
    rect_noise = (noise_1 * (inject_noise ** 2)) / (noise_2 - noise_1)
    return rect_noise


def fusion(rect_noise, noise_1):
    beta0 = 0.606
    beta1 = 0.394
    final_noise = beta0 * rect_noise + beta1 * noise_1
    return final_noise