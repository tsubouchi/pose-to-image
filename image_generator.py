import os
import base64
import logging
import tempfile
from PIL import Image
import requests
import io
from fal import client as fal

# Initialize detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flux API configuration
FLUX_API_KEY = "5db59d74-127a-4240-a028-2662d88522a4:f7e522e4afbf3486f03f771446bbfe4b"

def generate_image(pose_image, style_prompt, system_prompt):
    """
    Generate a new image using Flux Pro 1.1 based on the pose image and style
    """
    temp_files = []
    try:
        # Configure fal client
        fal.config({
            "credentials": FLUX_API_KEY
        })

        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_files.append(tmp_file.name)
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Input pose image saved to temporary file: {tmp_file.name}")

            # Read the image file as base64
            with open(tmp_file.name, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        logger.debug("Sending request to Flux Pro API")

        # Generate image using Flux Pro API
        result = fal.subscribe('fal-ai/flux-pro/v1.1-ultra', {
            'input': {
                'prompt': style_prompt,
                'negative_prompt': 'multiple people, bad anatomy, extra limbs, deformed hands, deformed fingers',
                'num_inference_steps': 30,
                'guidance_scale': 7.5,
                'controlnet_conditioning_scale': 1.0,
                'image_size': '1024x1024',
                'enable_safety_checker': False,
                'num_images': 1,
                'image': f"data:image/png;base64,{encoded_image}"
            },
            'logs': True,
            'onQueueUpdate': lambda update: logger.debug(f"Queue update: {update}")
        })

        if result and 'images' in result.data:
            # Get the first generated image URL
            image_url = result.data['images'][0]['url']
            logger.debug(f"Received image URL: {image_url}")

            # Download the image
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                image_data = image_response.content
                # Create PIL Image from the downloaded data
                img = Image.open(io.BytesIO(image_data))
                logger.debug(f"Successfully downloaded and opened generated image: format={img.format}, size={img.size}")
                return img
            else:
                raise ValueError(f"Failed to download generated image: {image_response.status_code}")
        else:
            raise ValueError("No image data received from Flux Pro")

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