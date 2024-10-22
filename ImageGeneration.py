import torch
from diffusers import StableDiffusionPipeline
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import base64
import io
import numpy as np
import os
import logging

# Load Stable Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load SAM2 model
sam_checkpoint = "C:/Users/Asus/Desktop/AI_Project/sam_vit_h.pth"
logging.info("Loading SAM model...")
sam = sam_model_registry["default"](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)
logging.info("SAM model loaded successfully")

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Directory for saving images
save_directory = "C:/Users/Asus/Desktop/AI_Project/saved_images"
os.makedirs(save_directory, exist_ok=True)

@app.post("/generate-image")
async def generate_image(prompt: str):
    # Generate the image from the Stable Diffusion model
    with torch.no_grad():
        image = pipe(prompt).images[0]

    # Save the generated image temporarily
    image_path = "generated_image.png"
    image.save(image_path)
    print(f"Image saved at: {image_path}")

    return FileResponse(image_path, media_type="image/png", filename="generated_image.png")


# Endpoint to analyze an existing image with CLIP
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Run CLIP analysis
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
    
    # Convert image features to a list or a JSON serializable format
    image_features = image_features.cpu().numpy().tolist()

    return {
        "clip_analysis": image_features
    }


# Predict segmentation mask using SAM2
@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    try:
        logging.info("Received image for segmentation")
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        logging.info(f"Image loaded successfully with shape {image_np.shape}")

        # Perform SAM segmentation
        logging.info("Starting SAM segmentation...")
        masks = mask_generator.generate(image_np)
        logging.info(f"SAM segmentation completed, found {len(masks)} masks")

        # Prepare mask as an image for visualization and saving
        mask_image = Image.fromarray(masks[0]['segmentation'].astype(np.uint8) * 255)  # Use the first mask
        mask_path = os.path.join(save_directory, "masked_image.png")
        mask_image.save(mask_path)
        logging.info(f"Masked image saved at {mask_path}")

        # Convert masks (NumPy arrays) to lists to make them JSON serializable
        masks_serializable = [{"segmentation": mask['segmentation'].tolist()} for mask in masks]

        return JSONResponse(content={
            "message": "Segmentation completed",
            "mask_path": mask_path,
            "masks": masks_serializable
        })

    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return JSONResponse(content={"message": "Error during segmentation", "error": str(e)}, status_code=500)

@app.get("/download-mask/")
async def download_mask():
    # Endpoint to download the last saved masked image
    mask_path = os.path.join(save_directory, "masked_image.png")
    if os.path.exists(mask_path):
        return FileResponse(mask_path, media_type="image/png", filename="masked_image.png")
    return JSONResponse(content={"message": "No mask found"}, status_code=404)

