import os

import cv2
import numpy as np
import glob as glob


def yuv2rgb(filename, width, height):
    with open(filename, 'rb') as f:
        nv12_data = f.read()

    # Create a YUV420p Mat object from the NV12 data
    yuv_frame = np.frombuffer(nv12_data, np.uint8).reshape((int(height * 1.5), width))

    # Convert YUV420p to BGR color space
    if filename.lower().endswith('.nv12'):
        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)

    elif filename.lower().endswith('.nv21'):
        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV21)

    rgb_img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return rgb_img


# img_list = []
# folder_path = r'C:\Users\wyq9893\Documents\Channel_based\noise'
# for filename in os.listdir(folder_path):
#     img_path = os.path.join(folder_path, filename)
#     img_list.append(img_path)
# for i, filename in enumerate(img_list):
#     width = 4096
#     height = 3060
#     rgb_img = yuv2rgb(filename, width, height)
#     # Display the image
#     cv2.imwrite(f'yuv2rgb_{i+1}.jpg', rgb_img)
