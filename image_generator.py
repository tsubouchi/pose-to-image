import google.generativeai as genai
from PIL import Image
from io import BytesIO

# Initialize the Gemini client
API_KEY = "AIzaSyDX8EkeJkVhsqK76SWz-S_euDYhV4gHGKU"
genai.configure(api_key=API_KEY)

def generate_image(pose_image):
    """
    Generate a new image using Gemini 2.0 Flash based on the pose image
    """
    try:
        # Convert pose image to bytes
        img_byte_arr = BytesIO()
        pose_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Prepare the prompt
        contents = (
            "Create an anime-style character image matching this pose. "
            "The character should be in a dynamic pose with a bright, detailed background. "
            "Use vibrant colors and maintain the same pose structure as shown in the reference."
        )

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content([contents, img_byte_arr])

        # Extract and return the generated image
        for part in response.parts:
            if hasattr(part, 'inline_data'):
                return Image.open(BytesIO(part.inline_data.data))

        raise ValueError("No image was generated")

    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")