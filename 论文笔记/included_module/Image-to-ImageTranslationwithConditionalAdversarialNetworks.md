# Image-to-Image Translation with Conditional Adversarial Networks

## 1. 简介：

本篇论文为 pix2pix 模型的论文，论文生成模型使用了Unet结构，判别模型使用了 70x70 patch结构，N x N patch是看这篇论文的主要目的。

## 2. N x N patch：

因为 L1 loss 可以有效地提取低频信息，所以，GAN中的判别器只需对于高频信息进行提取并做出比较即可，所以无需对于目标进行连续采样。文中使用卷积对于目标的每个区域进行采样，对于每个N x N 区域计算特征值，最后对所有区域的特征值取平均作为总的特征值。

N x N patch 中的 N x N 是指模型的输出层特征所对应的原始图像上的取样区域尺寸，不是最终的输出尺寸，可以通过输出尺寸和模型内的卷积层反推而出。