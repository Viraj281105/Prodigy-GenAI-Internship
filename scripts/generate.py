import torch
from diffusers import StableDiffusionPipeline

# This is the model ID for Stable Diffusion 1.5
model_id = "runwayml/stable-diffusion-v1-5"

# Load the pipeline. 
# torch_dtype=torch.float16 is CRITICAL for saving GPU memory (VRAM)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)

# Move the model to your GPU
pipe = pipe.to("cuda")

print("Model loaded. Ready to generate...")

# --- Your text prompt ---
prompt = "A high-resolution photo of an astronaut riding a horse on Mars"
# ---

# Generate the image
image = pipe(prompt).images[0]  

# Save the image to your project folder
image.save("astronaut_on_mars.png")

print("Success! Image saved as 'astronaut_on_mars.png'")