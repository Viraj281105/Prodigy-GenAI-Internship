import os
import glob
import cv2 # OpenCV
import re

# --- 1. Define Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, 'outputs')
output_video_path = os.path.join(script_dir, 'pix2pix_training_progress.mp4')

print(f"Looking for images in: {image_folder}")

# --- 2. Get and Sort Image Files ---
image_files_raw = glob.glob(os.path.join(image_folder, '*.png'))

if not image_files_raw:
    print(f"Error: No .png images found in {image_folder}")
else:
    def get_epoch_number(file_path):
        basename = os.path.basename(file_path)
        numbers = re.findall(r'\d+', basename)
        if numbers:
            return int(numbers[0])
        return -1
            
    image_files = sorted(image_files_raw, key=get_epoch_number)
    print(f"Found and sorted {len(image_files)} images.")

    # --- 3. Create the Video ---
    if image_files:
        # Read the first image to get dimensions
        first_frame = cv2.imread(image_files[0])
        height, width, layers = first_frame.shape

        # Define the video codec and create VideoWriter object
        # Using 'mp4v' for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        
        # Set FPS (Frames Per Second)
        # 40 images / 8 FPS = 5-second video
        fps = 8
        
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print("Creating video...")
        
        # Write all frames
        for img_path in image_files:
            frame = cv2.imread(img_path)
            video_writer.write(frame)
            
        # Add a 2-second pause on the final frame
        final_frame = cv2.imread(image_files[-1])
        for _ in range(fps * 2): # 8 FPS * 2 seconds = 16 extra frames
            video_writer.write(final_frame)

        # Release the video writer
        video_writer.release()
        print(f"\nSuccessfully saved Video to:\n{output_video_path}")
    else:
        print("Error: No image files found to create video.")