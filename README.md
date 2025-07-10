[Data Augmentation-aided fatigue delamination shape prognostics in composites]

# Installation
### Requirements
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)


### Notices:
* The augmentation file contains 5 generative models corresponding to the paper.
* The prediction file contains a BNN-based prognostic model.

# Data Availability
* The original C-scan data are provided in the paper.
* The KL matrices processed from the samples of the four original specimens are provided in the `dataset` folder.
* As for synthetic samples, only the GAN-generated samples under Case 1 are currently available.

# Citation
If you find this repo or our work useful for your research, please consider citing this paper and the papers below

a. Particle filter-based delamination shape prediction in composites subjected to fatigue loading;
b. Multiple local particle filter for high-dimensional system identification
