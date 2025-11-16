# PRDOIGY_GA_04: Image-to-Image Translation with Pix2Pix

This project is the 4th task for the Prodigy InfoTech Generative AI Internship. The objective is to implement **Pix2Pix**, a conditional Generative Adversarial Network (cGAN), to perform image-to-image translation.

We trained a model to translate segmentation maps of building facades into realistic photographs.

## 🚀 Final Results

After a 200-epoch training run, the model successfully learned to generate realistic photographs of building facades from their corresponding segmentation maps.

* **Input:** The "Input Image" (segmentation map)
* **Output:** The "Generated Image" (the model's prediction)
* **Ground Truth:** The "Real (Target) Image" (the original photo)


### Final Loss Graph

The final loss graph shows a healthy and stable training process.
* **G Loss (Generator):** This loss remains relatively high because it is dominated by the **L1 Loss** (multiplied by a `LAMBDA` of 100), which forces the generator to be pixel-accurate.
* **D Loss (Discriminator):** This loss remains low and stable, indicating the discriminator became a strong critic, which is necessary for generating high-quality, realistic images.


---

## 🛠️ Algorithm & Architecture

We implemented the **Pix2Pix** algorithm, which is a specific type of **Conditional Generative Adversarial Network (cGAN)**.

The core idea is to provide a "condition" (the input segmentation map) to both the generator and the discriminator, forcing the generator to create an image that is not just realistic, but a *plausible translation* of the input.

### 1. The Generator: A "U-Net"
The generator's job is to create the fake photo from the map. We used a **U-Net** architecture.

* **Encoder-Decoder:** The model first compresses the image down to a bottleneck (encoder) and then builds it back up (decoder).
* **Skip Connections:** The key feature of a U-Net. These connections link layers from the encoder directly to the decoder. This allows the model to pass **low-level information** (like edges and textures) directly, preventing them from being lost in the bottleneck. This is critical for generating sharp, detailed images.



### 2. The Discriminator: A "PatchGAN"
The discriminator's job is to determine if a (map, photo) pair is real or fake. Instead of a normal discriminator that outputs a single "real/fake" score for the whole image, we used a **PatchGAN**.

* **Patch-Based:** The PatchGAN classifies N x N "patches" of the image as real or fake.
* **Benefit:** This forces the generator to produce high-frequency details (like realistic textures) across the *entire* image, as any unrealistic "patch" will be penalized.

### 3. The Loss Functions
The model is trained by balancing two losses:

1.  **Adversarial Loss (`nn.BCEWithLogitsLoss`):** This is the standard GAN loss. It pushes the generator to create images that are indistinguishable from real photos.
2.  **L1 (MAE) Loss (`nn.L1Loss`):** This is the "conditioning" loss. It compares the generated photo to the real photo, pixel by pixel, and forces them to be structurally similar.

The final Generator Loss is a weighted sum:
`G_Loss = Adversarial_Loss + (LAMBDA * L1_Loss)`
(We used `LAMBDA = 100`, as recommended by the paper).

---

## ⚙️ Setup & Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/PRDOIGY_GA_04.git](https://github.com/your-username/PRDOIGY_GA_04.git)
    cd PRDOIGY_GA_04
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On Mac/Linux
    # source venv/bin/activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
