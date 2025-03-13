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
            "mime_type": "image/png",
            "data": image_data
        }

        # Create system prompt
        system_prompt = (
            "あなたは棒人間のポーズから日本のアニメスタイルのキャラクターを生成する専門家です。"
            "入力された画像は実写から抽出されたポーズで、カラーラインで表現されています。"
            "以下の要件に従って新しい画像を生成してください："
            "1. 棒人間で示された正確なポーズと比率を維持する"
            "2. 指定されたスタイルに合わせて詳細なキャラクター特徴を追加する"
            "3. キャラクターを引き立てる適切な背景を作成する"
            "4. 高品質で一貫性のある出力を確保する"
            "\n生成された画像はimagen3の品質で出力してください。"
        )

        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{style_prompt}"

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content([
            full_prompt,
            image_parts
        ])

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