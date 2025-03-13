import os
import base64
import logging
import tempfile
from PIL import Image
import requests
import io
import fal

# Initialize detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize fal client
fal.key = "5db59d74-127a-4240-a028-2662d88522a4:f7e522e4afbf3486f03f771446bbfe4b"

def generate_image(pose_image, style_prompt, system_prompt):
    """
    Generate a new image using Flux Pro 1.1 based on the pose image and style
    """
    temp_files = []
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_files.append(tmp_file.name)
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Input pose image saved to temporary file: {tmp_file.name}")

        # Create detailed prompt for Flux
        prompt = f"""
        {system_prompt}

        Style Specifications:
        {style_prompt}

        Follow these instructions precisely to create the image based on the reference pose.
        """

        logger.debug("Sending request to Flux Pro API")

        # Generate image using Flux Pro 1.1
        response = fal.run(
            'fal-ai/flux-pro:1.1-ultra',
            {
                'prompt': prompt,
                'negative_prompt': 'multiple people, bad anatomy, extra limbs, deformed hands, deformed fingers',
                'image_url': tmp_file.name,
                'num_inference_steps': 30,
                'guidance_scale': 7.5,
                'controlnet_conditioning_scale': 1.0,
                'width': 1024,
                'height': 1024,
                'seed': -1,  # Random seed
                'scheduler': 'euler_a'
            }
        )

        # Process the response
        if response and 'image' in response:
            # Save and verify the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as out_file:
                temp_files.append(out_file.name)
                image_data = base64.b64decode(response['image'])
                out_file.write(image_data)
                logger.debug(f"Saved decoded image to: {out_file.name}")

                # Verify the image can be opened
                img = Image.open(out_file.name)
                logger.debug(f"Successfully opened generated image: format={img.format}, size={img.size}")
                return img
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