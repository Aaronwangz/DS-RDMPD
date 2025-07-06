# DS-RDMPD

**A Dual-Stage Residual Diffusion Model with Perceptual Decoding for Remote Sensing Image Dehazing**

We present DS-RDMPD — a Dual-Stage Residual Diffusion Model with Perceptual Decoding for remote sensing image dehazing. DS-RDMPD features a coarse-to-fine dual-stage architecture that efficiently combines the strengths of a traditional UNet and a diffusion model. In the first stage, we introduce a Multi-channel Efficient Selective Synthesis UNet (MCESS-UNet) that performs initial dehazing and feature extraction using a multi-scale channel attention (MC) block and an Efficient Selective Synthesis (ESS) block. The output is then fed into a Residual Diffusion Model with Perceptual Decoding, which leverages a perceptual decoder to refine residual estimation and improve generative quality. Our method significantly reduces the computational cost typically associated with diffusion models, achieving high-quality restoration with only 300K training iterations and ~5 sampling steps. In addition to dehazing, DS-RDMPD also shows promising generalization in related tasks such as deraining and deblurring.

> ⚠️ The code is still under refinement and will be released soon.

## 📦 Available Resources

While the code is being finalized, you can access the following components:

- 🔹 **First-stage model weights**  
  [📥 Download](https://drive.google.com/drive/folders/1XWtq8Gn3MdlvIPw7_S750vFG7iy634AQ?usp=drive_link)

- 🔹 **Second-stage model weights**  
  [📥 Download](https://drive.google.com/drive/folders/1Q7PX3VwAymqgeB5IXvYIG3o7mdv3cFez?usp=drive_link)

- 🔹 **RSID dataset (used for training and evaluation)**  
  [📥 Download](https://drive.google.com/drive/folders/1abSw9GWyyOJINWCRNHBUoJBBw3FCttaS?usp=drive_link)

## 🙏 Acknowledgment

Our project is based on **[RDDM](https://github.com/nachifur/RDDM)**, and we are very grateful for this excellent work. Their contributions laid the foundation for our advancements in diffusion-based remote sensing image restoration.

---

Stay tuned for the full release, including training/inference code and detailed documentation. If you have any questions, please feel free to contact us at 3089777698qq.com
