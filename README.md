### Learned Pattern Similarity (LPS) for Multivariate Time Series Similarity

This is Matlab implementation of Learned Pattern Similarity introduced in our paper DAMI paper "Time series representation and similarity based on local autopatterns" in 2016. Note that there is an efficient R implementation introduced as an R package named LPStimeSeries on [CRAN](https://cran.r-project.org/web/packages/LPStimeSeries/) which works for univariate time series database where time series lengths are equal. Please also check [https://www.timeseriesclassification.com/](https://www.timeseriesclassification.com/) for java implementation. This version is also working on univariate time series.

LPStimeSeries is efficient as the main algorithm is coded in C however it cannot handle the univariate time series with varying lengths and the multivariate case. We also have MATLAB implementation in which the training is performed in a slightly different way (it is not exactly the same algorithm introduced in the paper) but the overall idea is similar. This repository has all necessary MATLAB functions and sample input necessary to run LPS on multivariate time series with varying lengths.

Note that main_LPS.m requires MTS data in specific Matlab format. You can check the structure of UWave.mat file uploaded under data folder. The input is a struct named as *mts* which has four attributes: *trainlabels*, *testlabels* which are arrays storing the label information. *train* and *test* as array of cells in which each cell is an MTS instance. 

References  
- Baydogan, M. G., & Runger, G. (2016). Time series representation and similarity based on local autopatterns. Data Mining and Knowledge Discovery, 30(2), 476-509.  
- Bagnall, A., Lines, J., Bostrom, A., Large, J., & Keogh, E. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data mining and knowledge discovery, 31(3), 606-660.
