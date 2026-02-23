import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# --- 1. Load the Model ---
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
print("Model loaded. Ready to generate...")

# --- 2. Set Up Your PROMPTS ---

# Let's try to force the model to make something specific.
# We want a beautiful, serene, blue flower.
prompt = "A single, (blue:1.5) flower, (serene:1.2), (beautiful:1.2), 8k, masterpiece, sharp focus, cinematic lighting"

# We'll also be very specific in our negative prompt.
negative_prompt = "ugly, (red:1.5), (yellow:1.5), blurry, grainy, bad art, cartoon, 3d render, (crowd:1.3), multiple flowers"

# --- 3. Set Parameters ---
g_scale = 8.0
steps = 40
seed = 98765 # A new seed for a new result

print(f"Generating image for: '{prompt}'")

# --- 4. Generate ---
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    num_inference_steps = steps,
    guidance_scale = g_scale,
    generator = generator
).images[0]

# --- 5. Save ---
image.save("weighted_prompt_image.png")
print("Successfully saved image to 'weighted_prompt_image.png'")