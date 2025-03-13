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
        base_prompt = """
        Create a high-quality image based on the provided pose reference.
        The pose must exactly match the stick figure reference, maintaining precise body positioning and proportions.

        Pose and Composition:
        - Follow the exact pose structure from the stick figure reference
        - Maintain natural body proportions and dynamic posing
        - Ensure anatomically correct articulation of joints and limbs
        - Create a balanced and visually appealing composition

        Technical Requirements:
        - Render in high resolution with sharp details
        - Use proper lighting and shadows for depth
        - Implement accurate perspective and depth
        - Pay attention to small details and textures
        """

        # Add style-specific prompt
        if "アニメ" in style_prompt:
            base_prompt += """
            Anime Style Specifications:
            - Modern Japanese anime art style with clean lines
            - Vibrant and harmonious color palette
            - Expressive eyes and facial features
            - Dynamic hair styling with natural flow
            - Detailed clothing with proper fabric folds
            - Maintain anime-specific proportions and aesthetics
            """
        else:
            base_prompt += """
            Photorealistic Style Specifications:
            - Highly detailed photorealistic rendering
            - Natural skin textures and features
            - Realistic fabric materials and textures
            - Professional photography lighting techniques
            - Subtle environmental reflections
            - Natural color grading and contrast
            """

        # Combine with style-specific details
        final_prompt = f"""
        {base_prompt}

        Style Details:
        {style_prompt}

        Additional Requirements:
        - Ensure the output matches the selected style perfectly
        - Create a cohesive scene with appropriate background
        - Add subtle environmental details that enhance the composition
        - Maintain professional quality standards throughout
        """

        logger.debug("Sending request to DALL-E API")

        # Generate image using DALL-E 3
        response = client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
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