import torch
from transformers import pipeline
from pathlib import Path

def main():
    # --- 1. SET UP PATHS ---
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent 
    model_path = base_dir / "gpt2-finetuned" / "final"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run 'train.py' first to create the model.")
        return

    # --- 2. DEFINE YOUR PROMPT ---
    prompt = "Holmes turned to me and said,"
    
    print(f"Loading fine-tuned model from: {model_path}")
    print(f"Using prompt: '{prompt}'")
    print("\n--- GENERATING TEXT (LOGIC-FOCUSED) ---")

    # --- 3. LOAD THE MODEL AND GENERATE ---
    try:
        generator = pipeline(
            'text-generation', 
            model=str(model_path), 
            tokenizer=str(model_path),
            device=0 if torch.cuda.is_available() else -1
        )

        # Generate the text
        output = generator(
            prompt, 
            max_new_tokens=75,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=generator.tokenizer.eos_token_id,

            # --- [ NEW SETTINGS FOR BETTER LOGIC ] ---
            # These are from your Hugging Face blog post!
            temperature=0.7,   # Makes the model's choices "safer" and less random.
            top_k=50,          # Restricts choices to the 50 most probable words.
            top_p=0.95         # Restricts choices based on cumulative probability.
        )

        # --- 4. PRINT THE RESULT ---
        print(output[0]['generated_text'])
        print("--------------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()