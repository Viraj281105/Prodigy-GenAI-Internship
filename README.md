# 🤖 Prodigy InfoTech - Generative AI Internship: Task 2

This project is a submission for Task 2 of the Prodigy InfoTech Generative AI Internship. The goal is to **utilize pre-trained generative models (like Stable Diffusion) to create images from text prompts.**

This repository contains a powerful, flexible Python script (`generate_cli.py`) that uses the Hugging Face `diffusers` library to run the **Stable Diffusion v1.5** model locally.



## ✨ Features

* **Command-Line Interface**: Easy to use from any terminal.
* **Prompt Weighting**: Use `(word:1.3)` syntax for fine-tuned control.
* **Negative Prompts**: Specify what *not* to generate.
* **Full Control**: Customize guidance scale, steps, and seed.
* **Batch Generation**: Create multiple images at once, saved as a grid.

---

## 🚀 Setup & Installation

To run this project locally, you'll need a PC with an NVIDIA GPU and CUDA.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Viraj281105/PRDOIGY_GA_02.git](https://github.com/Viraj281105/PRDOIGY_GA_02.git)
    cd PRDOIGY_GA_02
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install PyTorch with CUDA support:**
    (This example is for CUDA 12.1. Check [PyTorch.org](https://pytorch.org/get-started/locally/) for your specific version.)
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

4.  **Install required libraries:**
    ```bash
    pip install diffusers transformers accelerate
    ```

## 🎨 How to Use
The script generate_cli.py is your main tool. The only required argument is a prompt (-p).

### Basic Example
This generates a single image named cli_output.png.

```bash
python generate_cli.py -p "a beautiful photo of a golden retriever"
```
Advanced Example
```bash
This command generates a 4-image grid, using a detailed prompt with weighting, a negative prompt, 50 steps, and a specific seed.
```
```bash
python generate_cli.py -p "a (cyberpunk:1.3) city at night, (neon lights:1.2)" -n "daytime, blurry" -s 50 -num 4 -sd 12345 -f "cyberpunk_grid.png"
```

### All Arguments

| Flag | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `-p` | `--prompt` | **(Required)** | The main text prompt. |
| `-n` | `--negative_prompt` | (see script) | Concepts to avoid. |
| `-g` | `--scale` | 8.5 | Guidance scale (how strictly to follow prompt). |
| `-s` | `--steps` | 40 | Number of inference steps. |
| `-num`| `--num_images` | 1 | Number of images to generate. |
| `-sd`| `--seed` | -1 (random) | Seed for reproducible results. |
| `-f` | `--filename` | `cli_output.png`| Output filename. |

---

## 🖼️ Gallery: Generated Examples

Here are a few images generated using this script.

### Cyberpunk City
> **Prompt**: `"a (cyberpunk:1.3) city at night, (neon lights:1.2)" -n "daytime, blurry" -s 50 -num 4 -sd 12345`

![Cyberpunk City](cyberpunk_grid.png)

### Studio Ghibli Style
> **Prompt**: `"(Studio Ghibli style:1.3), a cozy artist's studio, (sunlight streaming through a window:1.1)" -n "realistic, photo" -s 40 -num 4 -sd 123`

![Studio Ghibli](ghibli_studio.png)

### Oil Painting
> **Prompt**: `"an (oil painting:1.4) of an old sea captain, (weathered face:1.3), (dramatic lighting:1.2)" -n "photo, 3d render" -s 50 -num 4 -sd 555`

![Captain Painting](captain_painting.png)