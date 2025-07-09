# A Dual-Stage Residual Diffusion Model with Perceptual Decoding for Remote Sensing Image Dehazing

abstract:Atmospheric pollutants, such as haze, severely affect the quality of remote sensing images, leading to blurred details and impairing their effectiveness in applications like environmental monitoring and agricultural resource management. In recent years, diffusion models have attracted widespread attention due to their powerful generative capabilities. \textcolor{red}{However, striking a balance between their expensive training costs and actual recovery effectiveness has become a major challenge. To address this key challenge, we propose a perceptual decoding dual-stage residual diffusion model (DS-RDMPD) for remote sensing image dehazing. The core innovation of our work lies in a dual-stage coarse-to-fine architecture that integrates the traditional UNet with a diffusion model, enabling efficient adaptation of diffusion-based restoration to the dehazing task. This design not only achieves strong performance but also demonstrates remarkable generalization across various image restoration scenarios.}
In the first stage, we use Multi-channel Efficient Selective Synthesis UNet (MCESS-UNet) to pre-process the remote sensing haze images. This architecture performs initial dehazing and feature extraction through a multi-scale channel attention (MC) block, and then performs enhanced spatial feature aggregation through an Efficient Selective Synthesis (ESS) block. The preprocessed image is then used as the conditional input of the Residual Diffusion Model with Perceptual Decoding, where the perceptual decoder improves the generation quality by further decoupling the condition to refine the residual estimate.
Extensive experiments on multiple datasets show that DS-RDMPD can achieve satisfactory results with only 300,000 iterations and about five sampling steps. It has achieved satisfactory results in both qualitative and quantitative experiments, and also performs well in rain removal and deblurring tasks, demonstrating the excellent generalization ability of the model.

优化一下。

## 🧠 Network Architecture

![Network Architecture](images/network_architecture.png)

---

### 1.🚀 Getting Started

We test the code on **PyTorch 1.13.0 + CUDA 11.7**.

### 2.Create a new conda environment

conda create -n DSRDMPD python=3.8
conda activate DSRDMPD
---

###  3.⚠️notice
The current open source code is less readable, but it can be trained and tested. You only need to modify the path. Note: modify the key image size parameters. We are currently accelerating the compilation of a more readable version.

## 4.📦 Available Resources

While the code is being finalized, you can access the following components:

- 🔹 **First-stage model weights**  
  [📥 Download](https://drive.google.com/drive/folders/1XWtq8Gn3MdlvIPw7_S750vFG7iy634AQ?usp=drive_link)

- 🔹 **Second-stage model weights**  
  [📥 Download](https://drive.google.com/drive/folders/1Q7PX3VwAymqgeB5IXvYIG3o7mdv3cFez?usp=drive_link)

- 🔹 **RSID dataset (used for training and evaluation)**  
  [📥 Download](https://drive.google.com/drive/folders/1abSw9GWyyOJINWCRNHBUoJBBw3FCttaS?usp=drive_link)

## 5.🙏 Acknowledgment

Our project is based on **[RDDM](https://github.com/nachifur/RDDM)**, and we are very grateful for this excellent work. Their contributions laid the foundation for our advancements in diffusion-based remote sensing image restoration.

---

Stay tuned for the full release, including training/inference code and detailed documentation. If you have any questions, please feel free to contact us at 3089777698qq.com
