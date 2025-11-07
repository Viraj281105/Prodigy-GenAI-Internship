# 🤖 PRDOIGY_GA_01: Fine-Tuning GPT-2 for Text Generation

This project is the first task for the Prodigy InfoTech Generative AI Internship. The goal is to fine-tune a pre-trained GPT-2 model on a custom dataset to generate coherent text that mimics the style and structure of the training data.

---

## Task Description
> Train a model to generate coherent and contextually relevant text based on a given prompt. Starting with GPT-2, a transformer model developed by OpenAI, you will learn how to fine-tune the model on a custom dataset to create text that mimics the style and structure of your training data.

## 🎯 Objective
The objective was to take the general-purpose **`gpt2`** model from Hugging Face and specialize it. I fine-tuned it on the text of **"The Adventures of Sherlock Holmes"** by Arthur Conan Doyle.

The result is a model that generates new text in the distinct, 19th-century descriptive style of the original stories.

## 🔧 Technologies Used
* Python
* PyTorch
* Hugging Face `transformers` library
* Hugging Face `datasets` library
* Hugging Face `accelerate` library

## 🚀 How to Use

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Viraj281105/PRDOIGY_GA_01.git](https://github.com/Viraj281105/PRDOIGY_GA_01.git)
    cd PRDOIGY_GA_01
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Model:**
    The dataset (Sherlock Holmes text) is already included in the `data/` folder. Simply run the training script:
    ```bash
    python scripts/train.py
    ```
    This will fine-tune the `gpt2` model and save the new, specialized model to the `/gpt2-finetuned/final` directory.

5.  **Generate New Text:**
    Once the model is trained, you can use the generation script to see the results.
    ```bash
    python scripts/generate.py
    ```
    You can change the `prompt` variable inside `scripts/generate.py` to get different results.

## 📊 Output Examples

This shows the clear difference between the base `gpt2` model and the fine-tuned version.

**Prompt:** `Holmes turned to me and said,`

---

### ❌ Before Fine-Tuning (Base GPT-2 Model)
The output is generic, grammatically correct, and sounds like a modern text or blog post.

> **Holmes turned to me and said,** "You know, I've been thinking about this for a while. It's a difficult situation, and I'm not sure what the right answer is. We need to consider all the options before we make a decision."

---

### ✅ After Fine-Tuning (Our Sherlock Model)
The output immediately adopts the 19th-century tone, references characters, and uses the correct narrative style.

> **Holmes turned to me and said,** “I shall be happy to take a cab to-night if you will.” He turned round and made a hurried step forward, but without turning his head to look. ‘Your Majesty’s business is to be done,‘he said he, and he rushed forward again. I bowed to him, with the utmost secrecy, for I knew