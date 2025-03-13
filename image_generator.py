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

def generate_image(pose_image, style_prompt):
    """
    Generate a new image using DALL-E 3 based on the pose image and style
    """
    temp_files = []
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_files.append(tmp_file.name)
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Input image saved to temporary file: {tmp_file.name}")

        # Create detailed prompt for DALL-E
        prompt = f"""
        Create a high-quality anime-style character illustration based on the provided pose reference.

        Pose and Composition:
        - Follow the exact pose structure from the stick figure reference
        - Maintain natural body proportions and dynamic posing
        - Ensure anatomically correct articulation of joints and limbs

        Character Style:
        - Modern Japanese anime art style with clean lines and bold colors
        - High attention to detail in facial features and expressions
        - Detailed clothing with proper fabric folds and textures
        - Dynamic hair styling with natural flow and movement

        Lighting and Atmosphere:
        - Use dramatic lighting to enhance the mood and depth
        - Implement proper shadows and highlights
        - Create a sense of depth with atmospheric perspective

        Technical Requirements:
        - Render in high resolution with sharp details
        - Maintain consistent line weights
        - Use proper color theory and shading techniques
        - Include subtle textures and material properties

        Style-specific details:
        {style_prompt}

        Additional Notes:
        - Create a cohesive and balanced composition
        - Ensure the character stands out from the background
        - Add subtle environmental details that complement the character
        - Maintain professional anime production quality standards
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