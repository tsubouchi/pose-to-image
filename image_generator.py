import google.generativeai as genai
from PIL import Image
from io import BytesIO
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
        contents = [{
            "parts": [
                {
                    "text": "この画像の構図で日本人の女子高生をimage3で出力して"
                },
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_data
                    }
                }
            ]
        }]

        logger.debug("Sending request to Gemini API")
        logger.debug(f"Request contents structure: {json.dumps(contents, indent=2)}")

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(contents)
        logger.debug("Received response from Gemini API")

        # Log response structure
        logger.debug(f"Response type: {type(response)}")
        logger.debug(f"Response parts count: {len(response.parts) if hasattr(response, 'parts') else 'No parts'}")

        # Process the response
        if response.parts:
            for i, part in enumerate(response.parts):
                logger.debug(f"Processing response part {i+1}")
                if hasattr(part, 'inline_data'):
                    try:
                        # Decode base64 data
                        response_data = part.inline_data.data
                        logger.debug(f"Response data length: {len(response_data)}")

                        image_bytes = base64.b64decode(response_data)
                        logger.debug(f"Decoded image bytes length: {len(image_bytes)}")

                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as out_file:
                            temp_files.append(out_file.name)
                            out_file.write(image_bytes)
                            logger.debug(f"Generated image saved to: {out_file.name}")

                            # Try to open and validate the image
                            try:
                                generated_image = Image.open(out_file.name)
                                logger.debug(f"Successfully opened image: format={generated_image.format}, size={generated_image.size}, mode={generated_image.mode}")
                                return generated_image
                            except Exception as img_error:
                                logger.error(f"Failed to open generated image: {img_error}")
                                raise Exception(f"Invalid image format: {img_error}")

                    except Exception as decode_error:
                        logger.error(f"Error processing generated image: {decode_error}")
                        raise Exception(f"Failed to process generated image: {decode_error}")

        raise ValueError("No image was generated in the response")

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