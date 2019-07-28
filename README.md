# Indoor Human Walking Path Reconstruction from a FMCW Radar Signal

## Introduction
Surveillance systems call for new solutions. Modern
video-cameras, in spite of high resolution performance, are
not able to cope with difficult conditions such as not lighted
environments and other blindness scenarios. On the contrary,
radar devices perform better in this kind of situations and
represent a valid alternative for the majority of surveillance
applications. In this paper we investigate how the tracking
procedure of a person who freely moves into two different
tested rooms could be improved considering both computational
efficiency and signal quality. We achieve this aim by using
a Gaussian Mixture Model in order to estimate the subject
position and its shape in the Range Doppler signal. Moreover, we
also developed further machine learning models to improve the
proposed algorithm and extend its application to a wide range
of demands, e.g. surveillance systems based on sensors networks
and person identification trough the use of further classification
tasks. We also investigate the use of a dimensionality reduction
system, i.e. several architectures of autoencoder models, which
can compress and decode the signal with varying loss of accuracy.
The proposed methods are discussed via experimental analyses
and graphical performance results, since they all belong to the
unsupervised learning field. We eventually consider our system
ready to provide solutions for real-world problems.

## Repository Description
Three jupyter notebooks describe the main steps of the analysis:
1. Preprocessing and Feature Extraction,
2. Spline and SOM,
3. Autoencoders.

In addition, three python files are used to store useful functions:
1. process.py,
2. denoising.py,
3. clustering.py.
