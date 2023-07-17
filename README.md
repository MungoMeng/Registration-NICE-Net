# NICE-Net: a Non-Iterative Coarse-to-finE registration Network for deformable image registration
In this study, we propose a Non-Iterative Coarse-to-finE registration Network (NICE-Net) for deformable registration. In the NICE-Net, we propose: (i) a Single-pass Deep Cumulative Learning (SDCL) decoder that can cumulatively learn coarse-to-fine transformations within a single pass (iteration) of the network, and (ii) a Selectively-propagated Feature Learning (SFL) encoder that can learn common image features for the whole coarse-to-fine registration process and selectively propagate the features as needed. Extensive experiments on six public datasets of 3D brain Magnetic Resonance Imaging (MRI) show that our proposed NICE-Net can outperform state-of-the-art iterative deep registration methods while only requiring similar runtime to non-iterative methods.  
**For more details, please refer to our paper. [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_9)] [[arXiv](https://arxiv.org/abs/2206.12596)]**

## Architecture
![architecture](https://github.com/MungoMeng/Registration-NICE-Net/blob/master/Figure/architecture.png)

## Publication
If this repository helps your work, please kindly cite our paper:
* **Mingyuan Meng, Lei Bi, Dagan Feng, Jinman Kim, "Non-iterative Coarse-to-fine Registration based on Single-pass Deep Cumulative Learning," International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), pp. 88-97, 2022, doi: 10.1007/978-3-031-16446-0_9. [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_9)] [[arXiv](https://arxiv.org/abs/2206.12596)]**
* **Mingyuan Meng, Lei Bi, Dagan Feng, Jinman Kim, "Brain Tumor Sequence Registration with Non-iterative Coarse-to-fine Networks and Dual Deep Supervision," International MICCAI Brainlesion Workshop (BrainLes), pp. 273–282, 2022, doi: 10.1007/978-3-031-33842-7_24. [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-33842-7_24)] [[arXiv](https://arxiv.org/abs/2211.07876)]**
