import google.generativeai as genai
from PIL import Image
import base64
import os
import tempfile
import logging
import json

# Initialize detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Gemini client
API_KEY = "AIzaSyDX8EkeJkVhsqK76SWz-S_euDYhV4gHGKU"
genai.configure(api_key=API_KEY)

def generate_image(pose_image, style_prompt):
    """
    Generate a new image using Gemini 2.0 Flash based on the pose image and style
    """
    temp_files = []
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_files.append(tmp_file.name)
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Input image saved to temporary file: {tmp_file.name}")

            # Read and encode the temporary file
            with open(tmp_file.name, 'rb') as img_file:
                image_bytes = img_file.read()
                logger.debug(f"Read {len(image_bytes)} bytes from input image")

                image_data = base64.b64encode(image_bytes).decode('utf-8')
                logger.debug(f"Image encoded to base64 (length: {len(image_data)})")

        # Create request contents
        parts = [
            {
                "text": """
                Generate a new image based on this stick figure pose.
                The output must be an image, not text.
                Return the response as image data in base64 format.

                Image generation requirements:
                - Use the exact pose from the input stick figure
                - Create an anime-style character
                - Follow these style guidelines:
                """ + style_prompt
            },
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": image_data
                }
            }
        ]

        logger.debug("Sending request to Gemini API")
        logger.debug(f"Request parts structure: {json.dumps(parts, indent=2)}")

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            parts,
            generation_config={
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40
            }
        )
        logger.debug(f"Response received - Type: {type(response)}")
        logger.debug(f"Response attributes: {dir(response)}")

        # Check all response data for debugging
        if hasattr(response, 'text'):
            logger.debug(f"Response text: {response.text}")

        if hasattr(response, 'candidates'):
            logger.debug(f"Response candidates: {response.candidates}")

        if not hasattr(response, 'parts'):
            logger.error("Response does not contain 'parts' attribute")
            raise ValueError("Invalid response format - no parts found")

        # Process response parts
        for i, part in enumerate(response.parts):
            logger.debug(f"Part {i} type: {type(part)}")
            logger.debug(f"Part {i} attributes: {dir(part)}")
            if hasattr(part, 'text'):
                logger.debug(f"Part {i} text: {part.text}")
            if hasattr(part, 'inline_data'):
                try:
                    logger.debug("Found inline_data in response")
                    image_data = part.inline_data.data

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

                except Exception as e:
                    logger.error(f"Error processing inline data: {e}")
                    continue

        logger.error("No valid image data found in response parts")
        raise ValueError("No image data in response")

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