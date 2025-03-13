import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64
import os
import tempfile
import logging

# Initialize the Gemini client
API_KEY = "AIzaSyDX8EkeJkVhsqK76SWz-S_euDYhV4gHGKU"
genai.configure(api_key=API_KEY)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_image(pose_image, style_prompt):
    """
    Generate a new image using Gemini 2.0 Flash based on the pose image and style
    """
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Temporary file created: {tmp_file.name}")

            # Read and encode the temporary file
            with open(tmp_file.name, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                logger.debug("Image successfully encoded to base64")

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            contents=[{
                "parts": [
                    {"text": "この画像の構図で日本人の女子高生をimage3で出力してください。"},
                    {
                        "mime_type": "image/png",
                        "data": image_data
                    }
                ]
            }]
        )

        logger.debug("Gemini API response received")

        # Extract and process the generated image
        if response.parts:
            for part in response.parts:
                if hasattr(part, 'inline_data'):
                    try:
                        # Save generated image to temporary file
                        image_bytes = base64.b64decode(part.inline_data.data)
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as out_file:
                            out_file.write(image_bytes)
                            logger.debug(f"Generated image saved to: {out_file.name}")

                            # Open and return the image
                            return Image.open(out_file.name)

                    except Exception as decode_error:
                        logger.error(f"Error processing generated image: {decode_error}")
                        raise Exception(f"Failed to process generated image: {decode_error}")

        raise ValueError("No image was generated in the response")

    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        raise Exception(f"Failed to generate image: {str(e)}")

    finally:
        # Cleanup temporary files
        if 'tmp_file' in locals():
            try:
                os.unlink(tmp_file.name)
                logger.debug("Temporary input file cleaned up")
            except:
                pass