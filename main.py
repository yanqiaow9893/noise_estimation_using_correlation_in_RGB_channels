# @Time    : 7/3/2023 11:03
# @Author  : wyq9893
# @File    : main.py
# @Software: PyCharm

import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from noiseEstimation import patch_noise_cal, patch_weight_cal, img_noise_cal, partition_image, random_patch_selection
from yuv2rgb import yuv2rgb


def regionOfInterest(img):
    # Define the coordinates of the area you want to extract (top-left and bottom-right points)
    top_left = (0, 500)  # (x, y) coordinates of the top-left corner
    bottom_right = (1400, 3000)  # (x, y) coordinates of the bottom-right corner

    # Extract the region of interest (ROI) from the input image
    extracted_roi = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Display the processed output image
    cv2.imwrite("roi.jpg", extracted_roi)
    return extracted_roi


if __name__ == '__main__':
    img_list = []
    folder_path = r'C:\Users\wyq9893\Documents\Noise Estimation\Channel_based\noise'
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img_list.append(img_path)
    # Define the standard deviation (sigma) for the Gaussian filter
    sigma = 5.0
    patch_size = 8

    overexposure_threshold = float(253/255) # Adjust this value as per your requirement
    underexposure_threshold = float(3/255)  # Adjust this value as per your requirement

    estimated_list = []
    name_list = []
    for img_path in img_list:
        img_name = os.path.basename(img_path)

        # Estimate the noise for image_path
        image = cv2.imread(img_path) # load image

        roi_image = regionOfInterest(image)
        # normalize the images by dividing them by 255
        original_img = np.array(image/255, dtype=float)

        # Apply Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(original_img, (0, 0), sigma)

        img_patches, blur_patches = partition_image(original_img, blurred_image, overexposure_threshold, underexposure_threshold, patch_size)

        # # Another way to partition the image
        # img_patches, blur_patches = random_patch_selection(original_img, blurred_image, 3000, overexposure_threshold, underexposure_threshold, patch_size)

        patch_noise = patch_noise_cal(img_patches)
        # gamma is a parameter that determines how strongly patches with high losses are filtered out
        patch_weight = patch_weight_cal(blur_patches, gamma=2.0)

        noise_level = img_noise_cal(patch_noise, patch_weight)  # it's the variance
        estimated_list.append(np.sqrt(noise_level))

        # # make the injected noise to be the same as the estimated noise level
        # injected_noise = np.sqrt(noise_level)
        #
        # # Injected the known noise to the original image to create a noisy image
        # new_noisy_img = noise_injection(original_img, injected_noise)
        # # print("Noise injected")
        # #
        # # Re-estimate the new noisy image
        # # Apply Gaussian blur to the image
        # new_noisy_img = np.array(new_noisy_img / 255, dtype=float)
        # new_noise = new_estimation(new_noisy_img, overexposure_threshold, underexposure_threshold)
        # rect_noise = rectification_linear_model(noise_level, new_noise, injected_noise)
        # true_noise = fusion(rect_noise, noise_level)
        # true_noise = np.sqrt(true_noise)
        # print("After rectification, noise level is ", true_noise)

        # estimated_list.append(true_noise)
        name_list.append(img_name)
        print(img_name, np.sqrt(noise_level))


    # format the estimated noise list
    estimated_list = ["%.6f" % num for num in estimated_list]
    # Create a DataFrame from the data
    df = pd.DataFrame({'File Name': name_list, 'Estimate Noise': estimated_list})

    # Specify the Excel file path and sheet name
    file_path = 'test.xlsx'
    sheet_name = 'noise level'

    # Write the DataFrame to Excel
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

    # Load the original image and the denoised image

    # # Create the first plot
    # fig1, ax1 = plt.subplots()
    # ax1.plot(name_list, estimated_list, 'bo-', label ='channel')
    # ax1.set_xlabel('image name')
    # ax1.set_ylabel('estimated noise level')
    # ax1.set_title('ISO801 Result')
    # ax1.legend()
    # plt.show()

