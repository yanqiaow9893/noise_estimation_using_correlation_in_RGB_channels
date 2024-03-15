# noise_estimation_using_correlation_in_RGB_channels

This code implements the method outlined in "Noise-Level Estimation from Single Color Image Using Correlations Between Textures in RGB Channels" by Akihiro Nakamura and Michihiro Kobayashi. The original paper can be found at: https://arxiv.org/abs/1904.02566.
![alt text](https://github.com/yanqiaow9893/noise_estimation_using_correlation_in_RGB_channels/blob/3c0f43faedc6fabd22e886668ccb9d4640c3f2e6/idea.png)

Additionally, the code supports '.NV12' and '.NV21' formats in addition to '.png' and '.jpg'. It incorporates a linear rectification model to reduce the estimation bias through the noise injection mechanism and thus improve the performance of noise estimation, where the rectification scheme idea is borrowed from "Noise Level Estimation for Natural Images Based on Scale-Invariant Kurtosis and Piecewise Stationarity" at: https://ieeexplore.ieee.org/document/7782836. 
