import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time
import math
import argparse  # <-- Import the library

# --- [ Helper Function: Image Grid (Unchanged) ] ---
def image_grid(imgs, rows, cols):
    if len(imgs) < rows * cols:
        w, h = imgs[0].size
        blank_image = Image.new('RGB', (w, h), (255, 255, 255))
        imgs.extend([blank_image] * (rows * cols - len(imgs)))

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# -----------------------------------------------------------------
# --- [ 1. SET UP COMMAND-LINE ARGUMENTS ] ---
# -----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")

# --- Define all the arguments ---
parser.add_argument(
    "-p", "--prompt",
    type=str,
    required=True,  # <-- This makes the prompt the only mandatory argument
    help="The main text prompt for generation."
)

parser.add_argument(
    "-n", "--negative_prompt",
    type=str,
    default="cartoon, blurry, low quality, drawing, painting, grainy, bad anatomy, ugly, watermark, text",
    help="The negative prompt to avoid concepts."
)

parser.add_argument(
    "-g", "--scale",
    type=float,
    default=8.5,
    help="Guidance scale (e.g., 8.5)"
)

parser.add_argument(
    "-s", "--steps",
    type=int,
    default=40,
    help="Number of inference steps (e.g., 40)"
)

parser.add_argument(
    "-num", "--num_images",
    type=int,
    default=1,  # <-- Default to 1 image for fast CLI use
    help="Number of images to generate (e.g., 4)"
)

parser.add_argument(
    "-sd", "--seed",
    type=int,
    default=-1,
    help="Seed for reproducibility (-1 for a random seed)"
)

parser.add_argument(
    "-f", "--filename",
    type=str,
    default="cli_output.png",
    help="Output filename (e.g., my_image.png)"
)

# --- Parse the arguments from the command line ---
args = parser.parse_args()

# -----------------------------------------------------------------
# --- [ 2. Load the Model ] ---
# -----------------------------------------------------------------
print("Loading Stable Diffusion model (v1.5)...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
print("Model loaded successfully.")


# -----------------------------------------------------------------
# --- [ 3. Configure the Generator ] ---
# -----------------------------------------------------------------
if args.seed == -1:
    # Use a random seed and get its value for printing
    generator = torch.Generator("cuda")
    actual_seed = torch.seed()
else:
    # Use the specified seed
    generator = torch.Generator("cuda").manual_seed(args.seed)
    actual_seed = args.seed

print(f"\n--- Generating {args.num_images} image(s) ---")
print(f"Prompt: {args.prompt}")
print(f"Negative: {args.negative_prompt}")
print(f"Seed: {actual_seed}")
print(f"Steps: {args.steps}, Scale: {args.scale}")


# -----------------------------------------------------------------
# --- [ 4. Run the Generation ] ---
# -----------------------------------------------------------------
start_time = time.time()
output = pipe(
    prompt = args.prompt,
    negative_prompt = args.negative_prompt,
    num_inference_steps = args.steps,
    guidance_scale = args.scale,
    num_images_per_prompt = args.num_images,
    generator = generator
)
end_time = time.time()

print(f"\nGeneration finished in {end_time - start_time:.2f} seconds.")

# -----------------------------------------------------------------
# --- [ 5. Save the Output ] ---
# -----------------------------------------------------------------
images = output.images

if args.num_images == 1:
    images[0].save(args.filename)
    print(f"Successfully saved image to '{args.filename}'")
else:
    cols = int(math.ceil(math.sqrt(args.num_images)))
    rows = int(math.ceil(args.num_images / cols))
    
    grid = image_grid(images, rows, cols)
    grid.save(args.filename)
    print(f"Successfully saved {args.num_images}-image grid to '{args.filename}'")