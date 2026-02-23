# 🎨 Neural Style Transfer (PRDOIGY_GA_05)

This project is a submission for the Prodigy InfoTech internship, Task 5. The goal is to implement a Neural Style Transfer (NST) script using PyTorch. The script applies the artistic style of one image (e.g., Van Gogh's "The Starry Night") to the content of another image (e.g., a photograph of a castle).

## Project Overview

Neural Style Transfer is a Generative AI technique that uses a pre-trained Convolutional Neural Network (CNN) to separate and recombine the "content" and "style" of images.

* **Content:** The high-level structure, objects, and layout of an image (e.g., "a castle").
* **Style:** The low-level textures, colors, and brushstroke patterns (e.g., "swirly, thick-painted lines").

The core idea, based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al., is to use the intermediate layers of a pre-trained network (like VGG-19) as a feature extractor. We then define two loss functions, **Content Loss** and **Style Loss**, and optimize the pixels of a new, generated image to minimize both simultaneously.

---

## 🏛️ Core Concepts

Before diving into the code, we established the "why" behind the key components.

### 1. The VGG-19 Model

* **Why VGG-19?** It's a deep CNN pre-trained on the ImageNet dataset. This training has taught it to recognize a rich hierarchy of features, from simple edges (early layers) to complex object shapes (deep layers). We use it as a pre-built "art critic" to *perceive* the images for us.
* **Why freeze the weights?** We **do not** train the VGG model. We freeze its parameters (`requires_grad=False`) and use it only as a fixed feature extractor. Our goal is to update the *pixels* of our generated image, not the "critic."
* **Why intermediate layers?**
    * **Content** is best captured by a single, deep layer (e.t., `conv_4`) that understands high-level object shapes.
    * **Style** is captured by the *correlation* of features across multiple layers, from low-level textures (`conv_1`, `conv_2`) to mid-level patterns (`conv_3`).

### 2. Content Loss

This loss function answers: "Does my generated image have the same *objects* as the content image?"
* **How it works:** We take the feature map from a deep VGG layer (e.g., `conv_4`) for both our *content image* and our *generated image*. The loss is simply the **Mean Squared Error (MSE)** between these two feature maps.
* **The Result:** Minimizing this loss forces the generated image to have the same high-level structure as the content photo.

### 3. Style Loss & The Gram Matrix

This loss function answers: "Does my generated image have the same *textures* and *patterns* as the style image?"
* **What is a Gram Matrix?** This is the key. A feature map has `C` channels. The Gram Matrix (size `C x C`) measures the **co-occurrence** (correlation) of these features. A high value at `Gram[i, j]` means "when I see feature `i` (e.g., 'vertical lines'), I also tend to see feature `j` (e.g., 'blue dabs') in the same areas."
* **How it captures "style"?** The Gram Matrix discards *where* the features are, only caring *that they co-occur*. This is the essence of style! Van Gogh's "style" isn't *where* the swirls are, but the fact that *swirls co-occur with dabs of yellow and blue*.
* **How it works:** We calculate the target Gram Matrix for the *style image* at several layers. We then calculate the Gram Matrix for our *generated image* at the same layers. The **Style Loss** is the MSE between these Gram Matrices.

---

## 📁 File Structure
PRDOIGY_GA_05/
├── images/
│   ├── neuschwanstein.jpg   (Content Image)
│   ├── starry_night.jpg     (Style Image)
│   └── output_image.jpg     (Final generated image)
│
├── scripts/
│   └── Style_Transfer_Task.ipynb  (The main Jupyter Notebook)
│
├── venv/
│   └── ... (Python virtual environment)
│
└── README.md                  (This file)

---

## 💻 Implementation: A Step-by-Step Walkthrough

Our implementation followed a 7-step process in a Jupyter Notebook (`Style_Transfer_Task.ipynb`).

### Step 1: Setup & Imports

* Imported `torch`, `torch.nn`, `torch.optim`, `PIL`, `matplotlib`, and `torchvision`.
* Set up the `device` (e.g., `cuda` or `mps`) for GPU acceleration.

### Step 2: Load Images & Preprocess

* Defined an `imsize` (e.g., 512px) for processing.
* Defined the VGG-19 normalization constants (mean and std).
* Created an `image_loader` function to open, resize (`transforms.Resize`), and convert images to tensors (`transforms.ToTensor`).
* Loaded the `style_img` and `content_img` and confirmed their shapes and devices.
* Created an `imshow` helper function to display tensors as images.

### Step 3: Load the VGG-19 Model

* Loaded the pre-trained `vgg19(weights=...).features`.
* Used `.to(device)` to move it to the GPU.
* Used `.eval()` to set it to evaluation mode.
* Iterated through all `param.requires_grad_(False)` to freeze the model.

### Step 4: Define the Loss Functions

* **`gram_matrix(input)`:** A function to calculate the Gram Matrix by reshaping the feature map and computing `torch.mm(features, features.t())`.
* **`ContentLoss(nn.Module)`:** A custom class that takes the `target` feature map in its constructor, `detach()`es it, and calculates `F.mse_loss(input, self.target)` in its `forward` method.
* **`StyleLoss(nn.Module)`:** A custom class that calculates and `detach()`es the *Gram Matrix* of the `target_feature` in its constructor. Its `forward` method computes the Gram Matrix of the `input` and calculates the `F.mse_loss` against the target Gram Matrix.
* Both modules are "transparent," returning their `input` tensor from the `forward` method.

### Step 5: Build the Style Transfer Model

* This was the most complex step. We created a `get_style_model_and_losses` function.
* It starts a new `nn.Sequential` model.
* **Normalization:** It first adds a custom `Normalization(nn.Module)` that uses the VGG mean and std. This bakes the normalization *into* the model.
* **Layer Iteration:** It loops through the `vgg19.children()`.
* **In-Place ReLU Bug:** We discovered `vgg19` uses `nn.ReLU(inplace=True)`, which breaks the backward pass. **The fix** was to check `if isinstance(layer, nn.ReLU)` and replace it with `nn.ReLU(inplace=False)`.
* **Loss Layer Injection:** As it iterates, it checks if the current layer's index matches our designated `content_layers` or `style_layers`. When it finds a match, it adds our custom `ContentLoss` or `StyleLoss` module to the model.
* **Trimming:** Finally, it trims the model, removing all VGG layers after the final loss module.

### Step 6: The Optimization Loop

* This is where we ran the main process. We defined the `input_img` (the image to be optimized) and an `optimizer` that *only* targets the `input_img`'s parameters (its pixels).
* The loop runs `num_steps` times, and in each step, it:
    1.  Calculates the `total_loss` (a weighted sum of `style_loss` and `content_loss`).
    2.  Calls `optimizer.zero_grad()`.
    3.  Calls `total_loss.backward()` to compute gradients.
    4.  Calls `optimizer.step()` to update the `input_img`'s pixels.

### Step 7: Post-processing & Display

* The `input_img` tensor is normalized. To view it, we created a `post_process` function.
* It moves the tensor to the `cpu()`, de-normalizes it (multiplying by `std_cpu` and adding `mean_cpu`), and uses `transforms.ToPILImage()` to convert it to a displayable image.
* **Device Bug:** We encountered a `RuntimeError` trying to de-normalize a `cpu` tensor with `cuda` stats. **The fix** was to create `mean_cpu` and `std_cpu` variables for the post-processing step.

---

## 🔬 The Optimization Journey: Tuning & Debugging

The biggest challenge was not the code, but the **hyperparameter tuning**. A "working" script produced a gray, blurry mess. We went through 9 rounds of tuning to find a good result.

* **Round 1 (L-BFGS, `style_weight=1e6`):**
    * **Result:** Faded, gray image. The castle was visible, but the style was almost non-existent.
    * **Diagnosis:** `style_weight` was too low. The optimizer prioritized keeping the content perfect.

* **Round 2 (L-BFGS, `style_weight=1e8`):**
    * **Result:** The optimization loop became unstable and the loss *exploded* (e.g., from 68 to 337).
    * **Diagnosis:** L-BFGS is sensitive to high-loss landscapes. It "overshot" the solution.

* **Round 3 (Adam, `input_img=noise`, `style_weight=1e6`):**
    * **Hypothesis:** Start from white noise, not the content image.
    * **Result:** A uniform gray-brown texture. The castle never "emerged."
    * **Diagnosis:** The `content_weight` (set to 1) was too weak to force the structure to appear from noise.

* **Round 4 (Adam, `input_img=content`, `style_weight=1e6`):**
    * **Result:** Back to the same faded image from Round 1.
    * **Diagnosis:** Adam is stable, but `style_weight=1e6` is still too low.

* **Rounds 5-7 (Adam, Content, `style_weight=1e8` to `1e10`):**
    * **Result:** The image *was* getting style (swirls!), but it was still gray, and the optimizer was unstable, with the loss jumping wildly.
    * **Diagnosis:** We were battling "exploding gradients." We fixed this by **lowering the learning rate** (e.g., to `0.001`), which made the loop stable. However, the image was *still* gray.

* **Round 8 (The Breakthrough):**
    * **Diagnosis:** We realized the *theoretical* problem. We were using `conv_1`...`conv_5` for style. This meant `conv_4` was being used for *both* content (castle shape) and style (swirl shape). The optimizer couldn't resolve this conflict.
    * **The Fix:** We changed our strategy.
        * **Content:** `conv_4` (high-level shapes)
        * **Style:** `conv_1`, `conv_2`, `conv_3` (low-level textures and colors *only*)

* **Round 9 (Final Attempt):**
    * **Hypothesis:** Combine our best ideas: New strategy (Round 8) + High `style_weight` (1e10) + Stable low `lr` (0.001).
    * **Result:** The loss was high but stable, and the image *finally* began to show both the castle's structure and the style's texture and color palette. This was deemed a successful completion of the task.

---

## 🖼️ Final Output

The final image represents a successful "compromise" by the optimizer, balancing the content structure of the castle with the textural style of the painting.

![Final Generated Image](./images/output_image.jpg)

*(Note: The "faded" look is a known limitation of the original NST algorithm. More advanced techniques, like color/histogram matching, are needed to get a 100% vibrant color transfer.)*

---

## 🚀 How to Run

1.  **Clone the repository.**
2.  **Set up the environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv
    # Activate it
    source venv/bin/activate  # (Linux/Mac)
    .\venv\Scripts\activate   # (Windows)
    # Install dependencies
    pip install torch torchvision matplotlib pillow jupyter
    ```
3.  **Register Jupyter Kernel (Optional but recommended):**
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=prodigy-task5
    ```
4.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
5.  **Run the notebook:** Open `scripts/Style_Transfer_Task.ipynb`. If you registered the kernel, select `Kernel > Change kernel > prodigy-task5`. Run all cells from top to bottom.