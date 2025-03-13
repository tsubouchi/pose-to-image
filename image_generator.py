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

        # システムプロンプト - 2段階処理
        system_prompt = """
        このプロセスは2段階で行います：

        1. まず、提供された棒人間の画像を解析してください。
        これは実写画像から抽出されたポーズデータで、人物の姿勢や動きを表現しています。

        2. 次に、解析したポーズに基づいて新しい画像を生成してください。
        以下の要件で画像を生成します：
        - imagen3の品質レベルで出力
        - 抽出されたポーズと完全に一致する姿勢
        - 現代的な日本の女子高生キャラクター
        - 自然な光と影の表現
        - 背景は日本の日常風景

        出力形式：高品質なイラスト画像
        """

        # Generate the image using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content([
            {
                "text": f"{system_prompt}\n\n{style_prompt}"
            },
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