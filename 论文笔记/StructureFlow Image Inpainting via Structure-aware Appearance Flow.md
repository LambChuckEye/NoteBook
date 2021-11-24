# StructureFlow: Image Inpainting via Structure-aware Appearance Flow

## 1. 简介：

本文提出了一种两段式的图像修复模型，先重建出缺失部分的结构信息，再重建出缺失部分的纹理信息。

相比于edge-connect，本文指出单纯的Canny边缘信息不足以覆盖全部结构信息，会造成大量的信息丢失，所以引入了edge-preserved smooth methods 来计算结构，可以去除高频纹理保留低频结构和边缘。

