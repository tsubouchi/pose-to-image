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
        contents = {
            "contents": [{
                "parts":[{
                    "text": "この棒人間のポーズに基づいて新しい画像を生成してください。以下の要素を含めてください：\n" + style_prompt
                },{
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_data
                    }
                }]
            }]
        }

        logger.debug("Sending request to Gemini API")
        logger.debug(f"Request contents structure: {json.dumps(contents, indent=2)}")

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(contents)
        logger.debug(f"Response received - Type: {type(response)}")
        logger.debug(f"Response attributes: {dir(response)}")

        # Process the response
        if hasattr(response, 'text'):
            logger.debug(f"Response text: {response.text}")

        if hasattr(response, 'candidates'):
            logger.debug(f"Response candidates: {response.candidates}")

        image_data = None
        if hasattr(response, 'parts'):
            for i, part in enumerate(response.parts):
                logger.debug(f"Part {i} type: {type(part)}")
                if hasattr(part, 'inline_data'):
                    image_data = part.inline_data.data
                    break

        if not image_data:
            raise ValueError("No image data found in response")

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