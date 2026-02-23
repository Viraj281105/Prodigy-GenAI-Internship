# 🧠 Prodigy InfoTech — Generative AI Internship

This repository contains all **5 tasks** completed during the **Prodigy InfoTech Generative AI Internship**, covering a range of generative AI techniques from classical statistical models to deep neural networks.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-99.5%25-orange)
![Python](https://img.shields.io/badge/Python-0.5%25-blue)
![GitHub stars](https://img.shields.io/github/stars/Viraj281105/Prodigy-GenAI-Internship?style=social)

---

## 🗂️ Repository Structure

```
Prodigy-GenAI-Internship/
├── Task_01_GPT2_Text_Generation/              # Fine-tune GPT-2 on Sherlock Holmes
├── Task_02_Stable_Diffusion_Text_to_Image/    # Text-to-image generation with Stable Diffusion
├── Task_03_Markov_Chain_Text_Generator/       # N-gram Markov Chain text generation
├── Task_04_Pix2Pix_Image_Translation/         # cGAN image-to-image translation
├── Task_05_Neural_Style_Transfer/             # Neural Style Transfer with VGG-19
└── README.md
```

---

## 📋 Tasks

---

### Task 01 — GPT-2 Fine-Tuning

Fine-tune GPT-2 on *The Adventures of Sherlock Holmes* to generate Victorian-style detective fiction text.

**Stack:** Python, PyTorch, Hugging Face Transformers

---

### Task 02 — Stable Diffusion Text-to-Image

Generate high-quality images from text prompts using Stable Diffusion. Explores prompt engineering and diffusion-based image synthesis.

**Stack:** Python, Diffusers, Hugging Face, PyTorch

---

### Task 03 — Markov Chain Text Generator

Statistical text generation using N-gram models across 3 iterations of increasing sophistication:

1. **Bigram** — simple 2-word chain transitions
2. **N-gram** — configurable higher-order chains
3. **Weighted N-gram** — probability-weighted sampling for more coherent outputs

**Stack:** Python (stdlib only — no external ML dependencies)

---

### Task 04 — Pix2Pix Image Translation

Conditional GAN (cGAN) implementation translating building facade segmentation maps into photorealistic images using the classic Pix2Pix architecture.

- **Generator:** U-Net encoder-decoder with skip connections
- **Discriminator:** PatchGAN for local realism scoring

**Stack:** Python, PyTorch

---

### Task 05 — Neural Style Transfer

Apply the artistic style of Van Gogh's *Starry Night* to a castle photograph using feature-level optimization over a pre-trained VGG-19 network.

- Content + style loss via Gram matrix comparison
- Iterative pixel-level optimization

**Stack:** Python, PyTorch, torchvision

---

## ⚖️ License

This project is licensed under the **MIT License**.

---

*Completed as part of the Prodigy InfoTech Generative AI Internship.*
