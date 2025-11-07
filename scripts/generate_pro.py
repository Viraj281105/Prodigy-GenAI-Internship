import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time
import math

# --- [ Helper Function: Image Grid ] ---
# This function takes a list of images and arranges them into a grid.
def image_grid(imgs, rows, cols):
    # If there are not enough images to fill the grid, fill with blank space
    if len(imgs) < rows * cols:
        print(f"Warning: Generating a {rows}x{cols} grid but only have {len(imgs)} images.")
        # Create a blank image
        w, h = imgs[0].size
        blank_image = Image.new('RGB', (w, h), (255, 255, 255))
        # Add blank images to the list to fill the grid
        imgs.extend([blank_image] * (rows * cols - len(imgs)))

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# --- [ 1. Load the Model ] ---
# This is done only once when the script starts.
print("Loading Stable Diffusion model (v1.5)...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
print("Model loaded successfully.")

# -----------------------------------------------------------------
# --- 🚀 YOUR SETTINGS ---
# ---
# (1) The main prompt. Use (word:weight) for emphasis.
prompt = "a (cinematic film still:1.2) of a (robot scientist:1.3) in a dark, futuristic lab, (glowing blue tubes:1.1), intricate details, 8k, masterpiece, dramatic lighting"

# (2) The negative prompt. Tell the model what to avoid.
negative_prompt = "cartoon, blurry, low quality, drawing, painting, grainy, bad anatomy, ugly, watermark, text"

# (3) Guidance Scale: How strictly to follow the prompt (7-12 is a good range).
g_scale = 8.5

# (4) Steps: How many steps to refine the image (30-50 is a good range).
steps = 40

# (5) Number of images to generate at once.
num_images = 4

# (6) Seed: The "random" starting point.
#    Set to -1 for a random seed every time.
#    Set to a specific number (e.g., 1024) for reproducible results.
seed = 1024 

# (7) Output filename.
filename = "my_generation.png"
# ---
# --- END OF SETTINGS ---
# -----------------------------------------------------------------


# --- [ 2. Configure the Generator ] ---
if seed == -1:
    # Use a random seed
    generator = torch.Generator("cuda")
else:
    # Use the specified seed for reproducible results
    generator = torch.Generator("cuda").manual_seed(seed)

print(f"\n--- Generating {num_images} image(s) ---")
print(f"Prompt: {prompt}")
print(f"Seed: {'random' if seed == -1 else seed}")

# --- [ 3. Run the Generation ] ---
start_time = time.time()
output = pipe(
    prompt = prompt,
    negative_prompt = negative_prompt,
    num_inference_steps = steps,
    guidance_scale = g_scale,
    num_images_per_prompt = num_images,
    generator = generator
)
end_time = time.time()

print(f"\nGeneration finished in {end_time - start_time:.2f} seconds.")

# --- [ 4. Save the Output ] ---
images = output.images

if num_images == 1:
    # Save the single image
    images[0].save(filename)
    print(f"Successfully saved image to '{filename}'")
else:
    # Auto-calculate grid size
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    
    # Create and save the grid
    grid = image_grid(images, rows, cols)
    grid.save(filename)
    print(f"Successfully saved {num_images}-image grid to '{filename}'")