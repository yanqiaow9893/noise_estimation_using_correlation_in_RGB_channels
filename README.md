# noise_estimation_using_correlation_in_RGB_channels

This code implements the method outlined in "Noise-Level Estimation from Single Color Image Using Correlations Between Textures in RGB Channels" by Akihiro Nakamura and Michihiro Kobayashi.
![alt text](https://github.com/yanqiaow9893/noise_estimation_using_correlation_in_RGB_channels/blob/3c0f43faedc6fabd22e886668ccb9d4640c3f2e6/idea.png)

Additionally, the code supports '.NV12' and '.NV21' formats in addition to '.png'. Note that the results work best on the RAW image rather than JPEG, see details in the original paper.
![alt text](https://github.com/yanqiaow9893/noise_estimation_using_correlation_in_RGB_channels/blob/a43555701b517e7ccaab38c04617cdd0cd50cd9e/result.png)

It incorporates a linear rectification model to reduce the estimation bias through the noise injection mechanism and thus improve the performance of noise estimation, where the rectification scheme idea is borrowed from "Noise Level Estimation for Natural Images Based on Scale-Invariant Kurtosis and Piecewise Stationarity" at: https://ieeexplore.ieee.org/document/7782836. 

# Reference
@ARTICLE{7782836,
  author={Dong, Li and Zhou, Jiantao and Tang, Yuan Yan},
  journal={IEEE Transactions on Image Processing}, 
  title={Noise Level Estimation for Natural Images Based on Scale-Invariant Kurtosis and Piecewise Stationarity}, 
  year={2017},
  volume={26},
  number={2},
  pages={1017-1030},
  abstract={Noise level estimation is crucial in many image processing applications, such as blind image denoising. In this paper, we propose a novel noise level estimation approach for natural images by jointly exploiting the piecewise stationarity and a regular property of the kurtosis in bandpass domains. We design a K-means-based algorithm to adaptively partition an image into a series of non-overlapping regions, each of whose clean versions is assumed to be associated with a constant, but unknown kurtosis throughout scales. The noise level estimation is then cast into a problem to optimally fit this new kurtosis model. In addition, we develop a rectification scheme to further reduce the estimation bias through noise injection mechanism. Extensive experimental results show that our method can reliably estimate the noise level for a variety of noise types, and outperforms some state-of-the-art techniques, especially for non-Gaussian noises.},
  keywords={Noise level;Estimation;Noise measurement;Image edge detection;Discrete cosine transforms;Optimization;Noise level estimation;scale invariant feature;kurtosis;piecewise stationarity},
  doi={10.1109/TIP.2016.2639447},
  ISSN={1941-0042},
  month={Feb},}
