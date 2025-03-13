import os
import base64
import logging
import tempfile
from PIL import Image
import requests
from openai import OpenAI
import io

# Initialize detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_image(pose_image, style_prompt, system_prompt):
    """
    Generate a new image using DALL-E 3 based on the pose image and style
    """
    temp_files = []
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_files.append(tmp_file.name)
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Input pose image saved to temporary file: {tmp_file.name}")

        # Create detailed prompt for DALL-E
        prompt = f"""
        System Instructions:
        {system_prompt}

        Reference Image Description:
        The provided image shows a stick figure pose that must be precisely replicated.
        This pose represents the exact body position, angles, and proportions that should be maintained.

        Key Focus Areas:
        1. Pose Accuracy:
        - Maintain exact joint angles and positions
        - Keep the same body orientation and tilt
        - Preserve limb positions and relationships
        - Match the overall pose composition

        2. Body Mechanics:
        - Ensure natural joint articulation
        - Maintain proper weight distribution
        - Keep realistic body proportions
        - Respect anatomical limitations

        3. Perspective and Depth:
        - Match the viewing angle of the stick figure
        - Maintain correct foreshortening
        - Preserve spatial relationships between body parts
        - Keep consistent perspective throughout

        Style Specifications:
        {style_prompt}

        Critical Requirements:
        - The output MUST match the stick figure pose exactly
        - Do not mirror or flip the pose
        - Maintain all spatial relationships between body parts
        - Ensure the character's gaze direction matches the pose
        """

        logger.debug("Sending request to DALL-E API")

        # Generate image using DALL-E 3
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="hd",
            response_format="b64_json"
        )

        # Process the response
        if response.data and len(response.data) > 0:
            image_data = response.data[0].b64_json
            logger.debug("Successfully received image data from DALL-E")

            # Save and verify the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as out_file:
                temp_files.append(out_file.name)
                img_bytes = base64.b64decode(image_data)
                out_file.write(img_bytes)
                logger.debug(f"Saved decoded image to: {out_file.name}")

                # Verify the image can be opened
                img = Image.open(out_file.name)
                logger.debug(f"Successfully opened generated image: format={img.format}, size={img.size}")
                return img
        else:
            raise ValueError("No image data received from DALL-E")

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