# noise_estimation_using_correlation_in_RGB_channels

This code implements the method outlined in "Noise-Level Estimation from Single Color Image Using Correlations Between Textures in RGB Channels" by Akihiro Nakamura and Michihiro Kobayashi.
![alt text](https://github.com/yanqiaow9893/noise_estimation_using_correlation_in_RGB_channels/blob/3c0f43faedc6fabd22e886668ccb9d4640c3f2e6/idea.png)

Additionally, the code supports '.NV12' and '.NV21' formats in addition to '.png'. Note that the results work best on the RAW image rather than JPEG, see details in the original paper.
![alt text](https://github.com/yanqiaow9893/noise_estimation_using_correlation_in_RGB_channels/blob/a43555701b517e7ccaab38c04617cdd0cd50cd9e/result.png)

It incorporates a linear rectification model to reduce the estimation bias through the noise injection mechanism and thus improve the performance of noise estimation, where the rectification scheme idea is borrowed from "Noise Level Estimation for Natural Images Based on Scale-Invariant Kurtosis and Piecewise Stationarity".

### References
- Nakamura, Akihiro, and Michihiro Kobayashi. "Noise-Level Estimation from Single Color Image Using Correlations Between Textures in RGB Channels." *arXiv preprint arXiv:1904.02566* (2019). https://arxiv.org/abs/1904.02566
- 
- Dong, Li, Zhou, Jiantao, and Tang, Yuan Yan. "Noise Level Estimation for Natural Images Based on Scale-Invariant Kurtosis and Piecewise Stationarity." *IEEE Transactions on Image Processing*, vol. 26, no. 2, 2017, pp. 1017-1030. https://doi.org/10.1109/TIP.2016.2639447.


