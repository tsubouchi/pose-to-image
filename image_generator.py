import os
import base64
import logging
import tempfile
from PIL import Image
import requests
import io
import google.generativeai as genai

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
STABILITY_KEY = os.getenv("STABILITY_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

def analyze_images_with_llm(pose_image: Image.Image, style_image: Image.Image):
    """
    Use Gemini to analyze both images and generate detailed descriptions
    """
    try:
        # Convert images to format compatible with Gemini
        pose_bytes = io.BytesIO()
        style_bytes = io.BytesIO()
        pose_image.save(pose_bytes, format='PNG')
        style_image.save(style_bytes, format='PNG')

        # Create prompt for image analysis
        prompt = """
        分析目的：
        1. ポーズ画像の姿勢とポーズの詳細な分析
        2. スタイル画像の視覚的特徴とアートスタイルの分析

        分析内容：
        1. ポーズ画像について：
        - 体の向きと姿勢
        - 手足の位置と角度
        - 全体的なバランス
        - 特徴的なポーズの要素

        2. スタイル画像について：
        - アートスタイルの特徴
        - 色使いとトーン
        - 線の使い方や質感
        - 光と影の表現
        - 独特の視覚効果

        出力形式：
        以下のセクションに分けて詳細に記述してください：
        - pose_details：ポーズの詳細な説明
        - style_elements：スタイル要素の説明
        - key_points：特に重要な要素のリスト
        - generation_tips：画像生成時の重要なポイント
        """

        # Get analysis from Gemini
        response = model.generate_content([
            prompt,
            pose_bytes.getvalue(),
            style_bytes.getvalue()
        ])

        return response.text

    except Exception as e:
        logger.error(f"Error in Gemini analysis: {str(e)}")
        return None

def generate_enhanced_prompt(analysis):
    """
    Use Gemini to generate an enhanced prompt based on the analysis
    """
    try:
        prompt_engineering = """
        以下の画像分析結果を元に、Stability AIのSDXL用の最適な生成プロンプトを作成してください。

        必要な要素：
        1. メインプロンプト
        - 正確なポーズの記述
        - スタイル要素の詳細
        - 画質向上のための技術的な指示
        - 重要な視覚効果の指定

        2. ネガティブプロンプト
        - 避けるべき要素
        - 品質低下を防ぐための指示

        3. 生成パラメータ
        - CFG Scale: 7-8の範囲
        - Steps: 20-30の範囲
        - Sampler: DPM++
        - Size: 512x768

        出力形式：
        1. メインプロンプト：高品質な画像生成のための詳細な指示
        2. ネガティブプロンプト：避けるべき要素のリスト
        3. パラメータ：最適な生成設定

        分析結果：
        {analysis}
        """

        response = model.generate_content(prompt_engineering)
        return response.text

    except Exception as e:
        logger.error(f"Error generating enhanced prompt: {str(e)}")
        return None

def generate_image_with_style(pose_image, style_image):
    """
    Generate a new image that combines the pose from pose_image with the style from style_image
    using multi-stage LLM processing and Stability AI
    """
    try:
        # Get detailed analysis from Gemini
        analysis = analyze_images_with_llm(pose_image, style_image)
        if not analysis:
            raise Exception("Failed to analyze images")

        # Generate enhanced prompt
        prompt_data = generate_enhanced_prompt(analysis)
        if not prompt_data:
            raise Exception("Failed to generate enhanced prompt")

        # Parse prompt data
        prompt_lines = prompt_data.split('\n')
        main_prompt = next((line for line in prompt_lines if line.startswith("メインプロンプト：")), "").replace("メインプロンプト：", "")
        negative_prompt = next((line for line in prompt_lines if line.startswith("ネガティブプロンプト：")), "").replace("ネガティブプロンプト：", "")

        # API endpoint for ultra generation
        host = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

        # Prepare headers
        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {STABILITY_KEY}"
        }

        # Send request with enhanced parameters
        response = requests.post(
            host,
            headers=headers,
            files={"none": ""},
            data={
                "prompt": main_prompt,
                "negative_prompt": negative_prompt,
                "output_format": "png",
                "cfg_scale": 7,
                "steps": 20,
                "sampler": "DPM++",
                "width": 512,
                "height": 768,
            }
        )

        if not response.ok:
            logger.error(f"API Response: {response.text}")
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        # Process response
        img = Image.open(io.BytesIO(response.content))
        logger.debug("Successfully generated styled image")

        return img

    except Exception as e:
        logger.error(f"Error in generate_image_with_style: {str(e)}")
        raise Exception(f"Failed to generate styled image: {str(e)}")


def generate_controlnet_openpose(pose_image, style_prompt):
    """
    Generate an image using Stability AI's latest API
    """
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pose_image.save(tmp_file.name, format='PNG')

        # API endpoint for ultra generation
        host = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {STABILITY_KEY}"
        }

        # Prepare request parameters
        data = {
            "prompt": style_prompt,
            "output_format": "png",
        }

        # Send request
        response = requests.post(
            host,
            headers=headers,
            files={"none": ""},
            data=data
        )

        if response.status_code == 404:
            logger.error(f"API endpoint not found: {host}")
            raise Exception("API endpoint not found. Please check the Stability AI documentation for the correct endpoint.")

        if not response.ok:
            logger.error(f"API Response: {response.text}")
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        # Process response
        img = Image.open(io.BytesIO(response.content))
        logger.debug(f"Successfully generated image using ultra API")

        return img

    except Exception as e:
        logger.error(f"Error in generate_controlnet_openpose: {str(e)}")
        raise Exception(f"Failed to generate image with ControlNet: {str(e)}")

    finally:
        # Cleanup temporary files
        try:
            os.unlink(tmp_file.name)
        except:
            pass


def generate_image(pose_image, style_prompt, system_prompt):
    """
    Generate a new image using Stability AI based on the pose image and style
    """
    temp_files = []
    try:
        # Save pose image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_files.append(tmp_file.name)
            pose_image.save(tmp_file.name, format='PNG')
            logger.debug(f"Input pose image saved to temporary file: {tmp_file.name}")

        # Prepare request parameters
        host = "https://api.stability.ai/v2beta/stable-image/control/sketch"

        files = {}
        params = {
            "prompt": style_prompt,
            "negative_prompt": "",
            "control_strength": 0.7,
            "seed": 0,
            "output_format": "png",
            "image": tmp_file.name
        }

        logger.debug("Sending request to Stability AI API")

        # Send generation request
        headers = {
            "Accept": "image/png",  # Changed from image/* to image/png
            "Authorization": f"Bearer {STABILITY_KEY}"
        }

        if params.get("image", None) is not None:
            files["image"] = open(params["image"], "rb")
            params.pop("image")

        response = requests.post(
            host,
            headers=headers,
            files=files,
            data=params
        )

        if not response.ok:
            logger.error(f"API Response: {response.text}")
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        # Decode response
        output_image = response.content
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")

        # Check for NSFW classification
        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")

        # Save and return result
        img = Image.open(io.BytesIO(output_image))
        logger.debug(f"Successfully received and opened generated image: format={img.format}, size={img.size}")
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