# @Time    : 7/3/2023 11:05
# @Author  : wyq9893
# @File    : noiseEstimation.py
# @Software: PyCharm

import numpy as np
import cv2
import random
from numba import jit



def is_patch_exposed(patch, overexposure_threshold, underexposure_threshold):
    # Calculate the average pixel intensity of the patch
    avg_intensity = np.mean(patch)

    # Check if the patch is overexposed or underexposed
    if avg_intensity > overexposure_threshold or avg_intensity < underexposure_threshold:
        return True
    else:
        return False



def partition_image(img, blur_img, patch_size):
    """
    uniformly partition into pxp non-overlapping square patches
    :param image: the noisy image
    :param patch_size: p x p
    :return: p x p non-overlapping square patches
    """
    img_height, img_width = img.shape[:2]

    # Calculate the number of patches in each dimension
    v_num = img_height // patch_size
    h_num = img_width // patch_size

    patches = []
    blur_patches = []

    for j in range(v_num):
        for i in range(h_num):
            y = j * patch_size
            x = i * patch_size
            patch = img[y:y + patch_size, x:x + patch_size]

            if not is_patch_exposed(patch, Th_over, Th_under):
                patches.append(patch)
                blur_patch = blur_img[y:y + patch_size, x:x + patch_size]
                blur_patches.append(blur_patch)

    return patches, blur_patches


def random_patch_selection(img, blur_img, Np, Th_over, Th_under, patch_size):
    """

    :param img: the original noisy image
    :param blur_img: the gaussian blurred image
    :param Np: Number of patches to sample
    :param Th_over: overestimated threshold
    :param Th_under: underestimated threshold
    :param patch_size: the specified patch size
    :return: image_patches: all the random sampled patch Pi (i=1, 2,..., Np)
    :return :blur_patches: all the random sampled blurry patch Pi
    """
    # Randomly sampled image patch Pi
    image_height, image_width, _ = img.shape  # Get image dimensions

    patch_coordinates = []  # Generate random coordinates for the top-left corner of each patch
    for _ in range(Np):
        x = random.randint(0, image_width - patch_size)
        y = random.randint(0, image_height - patch_size)
        patch_coordinates.append((x, y))

    # Extract patches using the coordinate and patch size
    image_patches = []
    blur_patches = []

    for x, y in patch_coordinates:
        # get the patches of the original image
        patch = img[y:y + patch_size, x:x + patch_size]

        # Check if the patch is exposed
        if not is_patch_exposed(patch, Th_over, Th_under):
            # if the patch is valid, then attach the patch to image_patches
            image_patches.append(patch)
            # Use the same x and y coordinates to get the blur_patch
            blur_patch = blur_img[y:y + patch_size, x:x + patch_size]
            blur_patches.append(blur_patch)

    return image_patches, blur_patches


def patch_noise_cal(img_patch):
    """

    :param img_patch: image patch Pi (i =1, 2, ...Np)
    :return: estimate_noise_list: noise level estimation for each patch Pi
    """
    # img_patch shape: height, width, channel
    estimate_noise_list = []
    for pi in img_patch:
        # Separate the channels
        Ir = pi[:, :, 0]  # Red channel
        Ig = pi[:, :, 1]  # Green channel
        Ib = pi[:, :, 2]  # Blue channel

        # Calculate mean of variance, a list of the mean of the channelwise variance of pixel values
        alpha = (np.var(Ir) + np.var(Ig) + np.var(Ib)) / 3
        # Calculate variance of mean for a single patch
        beta = np.var(np.add(np.add(Ir, Ig), Ib) / 3)
        # noise level estimate for each patch Pi. if condition C is satisfied
        estimate_noise = 3 / 2 * (alpha-beta)

        estimate_noise_list.append(abs(estimate_noise))

    return estimate_noise_list


def img_noise_cal(patch_noise, patch_weight):
    """

    :param patch_noise: Nx1 array of estimated noise for patch Pi
    :param patch_weight: Nx1 array of weight wi for patch Pi
    :return: noise level estimate of the entire image
    """
    res = [patch_weight[i] * patch_noise[i] for i in range(len(patch_noise))]
    nomi = np.sum(res)
    deno = np.sum(patch_weight)
    noise_level = nomi / deno

    return noise_level


def patch_weight_cal(blur_patch, gamma):
    """

    :param blur_patch:  blur image patch Pi (i =1, 2, ...Np)
    :param gamma: how strongly patches with high losses are filtered out
    :return: weight_list: list of the patch weight
    """
    sum_loss = 0
    Loss_list = []

    for pi in blur_patch:
        Ir = pi[:, :, 0]
        Ig = pi[:, :, 1]
        Ib = pi[:, :, 2]
        Loss_i = (np.var(Ir - Ig) + np.var(Ig - Ib) + np.var(Ib - Ir)) / 3
        sum_loss += Loss_i

        Loss_list.append(Loss_i)

    weight_list = []

    normalize_factor = sum_loss / len(blur_patch)

    # if normalize_factor is zero, meaning the patch has pixel value the same in R,G,B channels respectively
    if normalize_factor == 0:
        for i in range(len(blur_patch)):
            weight = 1
            weight_list.append(weight)
    else:
        for i in range(len(blur_patch)):
            weight = np.exp(-gamma * Loss_list[i] / normalize_factor)
            weight_list.append(weight)

    return weight_list

