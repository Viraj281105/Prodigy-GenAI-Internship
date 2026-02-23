import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from pathlib import Path

def main():
    # --- 1. DEFINE PATHS ---
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent 
    dataset_path = base_dir / "data" / "my_data.txt"
    output_dir = base_dir / "gpt2-finetuned"
    model_name = "gpt2"

    # --- 2. LOAD TOKENIZER & MODEL ---
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # --- 3. LOAD AND PREPARE DATASET ---
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    print(f"Loading and tokenizing dataset from: {dataset_path}")
    dataset = load_dataset('text', data_files={'train': str(dataset_path)})

    # --- [ THE FIX ] ---
    # Filter out empty or whitespace-only lines
    print("Filtering out empty lines from the dataset...")
    dataset = dataset.filter(lambda example: example['text'] is not None and len(example['text'].strip()) > 0)
    # --- [ END OF FIX ] ---

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128
        )

    # Apply tokenization
    print("Tokenizing the filtered dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # --- 4. DEFINE TRAINING ARGUMENTS ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=3,                
        per_device_train_batch_size=4,     
        save_steps=10_000,                 
        save_total_limit=2,                
        logging_steps=500, # Log loss every 500 steps
    )

    # --- 5. INITIALIZE THE DATA COLLATOR ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # --- 6. CREATE THE TRAINER ---
    trainer = Trainer(
        model=model,                       
        args=training_args,                
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
    )

    # --- 7. START TRAINING! ---
    print("--- Starting Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")

    # --- 8. SAVE THE FINAL MODEL & TOKENIZER ---
    final_model_path = output_dir / "final"
    print(f"Saving final model to {final_model_path}...")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print("Done!")

if __name__ == "__main__":
    main()