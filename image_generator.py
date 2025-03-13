import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64
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
        # Convert pose image to base64
        img_byte_arr = BytesIO()
        pose_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        image_data = base64.b64encode(img_bytes).decode('utf-8')

        logger.debug("Image data encoded successfully")

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

        # Extract and return the generated image
        if response.parts:
            for part in response.parts:
                if hasattr(part, 'inline_data'):
                    try:
                        image_bytes = base64.b64decode(part.inline_data.data)
                        logger.debug("Successfully decoded image data")

                        # Try to identify the image format
                        img = Image.open(BytesIO(image_bytes))
                        img_format = img.format
                        logger.debug(f"Image format identified: {img_format}")

                        return img
                    except Exception as decode_error:
                        logger.error(f"Error decoding image data: {decode_error}")
                        raise Exception(f"Failed to decode generated image: {decode_error}")

        raise ValueError("No image was generated in the response")

    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        raise Exception(f"Failed to generate image: {str(e)}")