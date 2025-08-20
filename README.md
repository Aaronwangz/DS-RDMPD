# 【TGRS 2025】A Dual-Stage Residual Diffusion Model with Perceptual Decoding for Remote Sensing Image Dehazing

This is the official PyTorch implementation of the paper:

> **A Dual-Stage Residual Diffusion Model with Perceptual Decoding for Remote Sensing Image Dehazing**  
> Hao Zhou, Yalun Wang, Qian Zhang, Tao Tao, and Wenqi Ren  
> *IEEE Transactions on Geoscience and Remote Sensing*, 2025  
> [Paper Link](https://ieeexplore.ieee.org/document/11130517)

We combined the traditional UNet model with the diffusion model to propose a two-stage network architecture, DS-RDMPD. This model achieved satisfactory results in remote sensing dehazing, real-world raindrop removal, and real-world denoising, demonstrating strong generalization capabilities. The paper can be found in the link above.

---

## 🧠 Network Architecture

![Network Architecture](images/1.png)

![Network Architecture](images/thin.jpg)![Network Architecture](images/moderate.jpg)![Network Architecture](images/thick.jpg)![Network Architecture](images/rain.jpg)![Network Architecture](images/blur.jpg)

---

### 🚀 Getting Started

We train and test the code on **PyTorch 1.13.0 + CUDA 11.7**. The detailed configuration is mentioned in the paper.

### Create a new conda environment
<pre lang="markdown">conda create -n DSRDMPD python=3.8 
conda activate DSRDMPD  </pre>

###  ⚠️ notice
Remember to modify the path to the dataset before running the test and training code. Different image resolutions require modifying the relevant parameter parameters.

## 📦 Available Resources

While the code is being finalized, you can access the following components:

- 🔹 **First-stage model weights**  
  [📥 Download](https://drive.google.com/drive/folders/1XWtq8Gn3MdlvIPw7_S750vFG7iy634AQ?usp=drive_link)

- 🔹 **Second-stage model weights**  
  [📥 Download](https://drive.google.com/drive/folders/1Q7PX3VwAymqgeB5IXvYIG3o7mdv3cFez?usp=drive_link)

- 🔹 **RSID dataset (used for training and evaluation)**  
  [📥 Download](https://drive.google.com/drive/folders/1abSw9GWyyOJINWCRNHBUoJBBw3FCttaS?usp=drive_link)

---
## 📖 Citation
If you find our work helpful in your research, please consider citing it. We appreciate your support！😊
<pre lang="markdown"> 
@ARTICLE{11130517,
  author={Zhou, Hao and Wang, Yalun and Zhang, Qian and Tao, Tao and Ren, Wenqi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Dual-Stage Residual Diffusion Model with Perceptual Decoding for Remote Sensing Image Dehazing}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Remote Sensing Image Dehazing;Diffusion Model;Computer Vision;Multi-Scale Channel Attention},
  doi={10.1109/TGRS.2025.3600540}
  }
  </pre>
---
## 🙏 Acknowledgment 

Our project is based on **[RDDM](https://github.com/nachifur/RDDM)**, and we are very grateful for this excellent work. Their contributions laid the foundation for our advancements in diffusion-based remote sensing image restoration.

---
If you have any questions, please feel free to contact us at aaron@ahut.edu.cn
