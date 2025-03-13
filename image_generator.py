import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64

# Initialize the Gemini client
API_KEY = "AIzaSyDX8EkeJkVhsqK76SWz-S_euDYhV4gHGKU"
genai.configure(api_key=API_KEY)

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

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            contents=[
                "この画像の構図で日本人の女子高生をimage3で出力してください。",
                {
                    "mime_type": "image/png",
                    "data": image_data
                }
            ]
        )

        # Extract and return the generated image
        if response.parts:
            for part in response.parts:
                if hasattr(part, 'inline_data'):
                    image_bytes = base64.b64decode(part.inline_data.data)
                    return Image.open(BytesIO(image_bytes))

        raise ValueError("No image was generated in the response")

    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")