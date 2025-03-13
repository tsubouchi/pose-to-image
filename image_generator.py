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
        image_parts = {
            "text": "この棒人間のポーズに基づいて、新しい画像を生成してください。",
            "inline_data": {
                "mime_type": "image/png",
                "data": image_data
            }
        }

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            contents=[
                {
                    "text": "この棒人間のポーズで日本人の女子高生をimagen3で出力してください。"
                    "\n以下の要素を含めてください："
                    "\n- 明るく現代的なアニメスタイル"
                    "\n- 自然な光と影の表現"
                    "\n- 背景は日本の学校や街並み"
                    f"\n\n追加のスタイル指定：{style_prompt}"
                },
                image_parts
            ],
            generation_config={
                "temperature": 0.4,
                "image_quality": "high"
            }
        )

        # Extract and return the generated image
        if response.parts:
            for part in response.parts:
                if hasattr(part, 'inline_data'):
                    image_bytes = base64.b64decode(part.inline_data.data)
                    generated_image = Image.open(BytesIO(image_bytes))
                    return generated_image

        raise ValueError("No image was generated in the response")

    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")