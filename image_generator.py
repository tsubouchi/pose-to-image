import os
import base64
import logging
import tempfile
from PIL import Image
import requests
import io
from datetime import datetime

# Initialize detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Stability API configuration
STABILITY_KEY = "sk-Rm5frm48K7sArubPQgJ9r2w8Q75XH5y0UR215NC4Fjndu7gz"
STABILITY_API_HOST = "https://api.stability.ai/v2beta/generation/image-to-image"

def generate_image(pose_image, style_prompt, system_prompt):
    """
    Generate a new image using Stability AI based on the pose image and style
    """
    temp_files = []
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_files.append(tmp_file.name)
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Input pose image saved to temporary file: {tmp_file.name}")

        # Prepare request parameters
        params = {
            "prompt": style_prompt,
            "image": tmp_file.name,
            "num_images": 1,
            "image_strength": 0.35,
            "steps": 30,
            "cfg_scale": 7.5,
            "sampler": "K_EULER_ANCESTRAL"
        }

        logger.debug("Sending request to Stability AI API")

        # Send generation request
        response = send_generation_request(STABILITY_API_HOST, params)

        if response.status_code == 200:
            # Process the response
            image_data = response.content
            img = Image.open(io.BytesIO(image_data))
            logger.debug(f"Successfully received and opened generated image: format={img.format}, size={img.size}")
            return img
        else:
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        raise Exception(f"Failed to generate image: {str(e)}")

    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {temp_file}: {cleanup_error}")

def send_generation_request(host, params, files=None):
    """Helper function to send request to Stability AI API"""
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    # Send request
    logger.debug(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )

    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response