# NICE-Net: Non-Iterative Coarse-to-finE registration Network for deformable image registration
In this study, we propose a Non-Iterative Coarse-to-finE registration Net-work (NICE-Net) for deformable registration. In the NICE-Net, we propose: (i) a Single-pass Deep Cumulative Learning (SDCL) decoder that can cumulatively learn coarse-to-fine transformations within a single pass (iteration) of the network, and (ii) a Selectively-propagated Feature Learning (SFL) encoder that can learn common image features for the whole coarse-to-fine registration process and selectively propagate the features as needed. Exten-sive experiments on six public datasets of 3D brain Magnetic Resonance Imaging (MRI) show that our proposed NICE-Net can outperform state-of-the-art iterative deep registration methods while only requiring similar runtime to non-iterative methods.
**For more details, please refer to our paper.**

## Architecture
![architecture](https://github.com/MungoMeng/Registration-NICE-Net/blob/master/Figure/architecture.png)

## Publication
If this repository helps your work, please kindly cite our paper:
* **Mingyuan Meng, Lei Bi, David Dagan Feng, Jinman Kim, "Non-iterative Coarse-to-fine Registration based on Single-pass Deep Cumulative Learning," International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2022.**
