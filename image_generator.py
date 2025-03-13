import os
import base64
import logging
import tempfile
from PIL import Image
import requests
import io
import google.generativeai as genai
import json

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
        目的：2枚の画像を分析し、1枚目のポーズを2枚目の画風で再現するための詳細情報を抽出

        1. ポーズ画像の分析：
        - 体全体の姿勢とポーズの詳細
        - 手足の位置と角度
        - 頭の向きと表情
        - 体の捻りや重心
        - 特徴的なジェスチャーや動き

        2. スタイル画像の分析：
        - アートスタイルの特徴（アニメ、リアル等）
        - 色使いとカラーパレット
        - 線の質感と太さ
        - シェーディングと陰影の付け方
        - 特徴的な視覚効果
        - 背景の処理方法

        出力形式：
        {
          "pose_details": "ポーズの詳細な説明",
          "style_elements": "画風の特徴",
          "composition": "構図とフレーミング",
          "key_points": ["重要な要素のリスト"],
          "technical_aspects": "技術的な詳細"
        }
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
        prompt_engineering = f"""
        以下の画像分析結果を元に、Stability AIのSDXL用の最適な生成プロンプトを作成してください。

        分析結果：
        {analysis}

        プロンプトの要件：
        1. メインプロンプト
        - "masterpiece, best quality" で始める
        - ポーズの正確な記述（姿勢、手足の位置、表情）
        - スタイル要素の詳細（画風、色調、線の特徴）
        - 技術的な品質指定（解像度、シャープネス等）

        2. ネガティブプロンプト
        - 避けるべき要素（低品質、ブレ、歪み等）
        - ポーズが崩れる要因の排除

        3. 生成パラメータ
        - CFG Scale: 7
        - Steps: 20
        - Size: 512x768

        出力形式：
        {
          "main_prompt": "メインプロンプト",
          "negative_prompt": "ネガティブプロンプト",
          "parameters": {
            "cfg_scale": 7,
            "steps": 20
          }
        }
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
        logger.info("Analyzing images with Gemini...")
        analysis = analyze_images_with_llm(pose_image, style_image)
        if not analysis:
            raise Exception("Failed to analyze images")

        # Generate enhanced prompt
        logger.info("Generating enhanced prompt...")
        prompt_data = generate_enhanced_prompt(analysis)
        if not prompt_data:
            raise Exception("Failed to generate enhanced prompt")

        # Parse prompt data
        try:
            prompt_data = json.loads(prompt_data)
        except:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse JSON, using text extraction")
            lines = prompt_data.split('\n')
            prompt_data = {
                "main_prompt": next((l for l in lines if "masterpiece" in l), "masterpiece, best quality"),
                "negative_prompt": next((l for l in lines if "low quality" in l), "NSFW, low quality"),
                "parameters": {"cfg_scale": 7, "steps": 20}
            }

        # API endpoint for ultra generation
        host = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

        # Prepare headers
        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {STABILITY_KEY}"
        }

        # Send request with enhanced parameters
        logger.info("Sending request to Stability AI...")
        response = requests.post(
            host,
            headers=headers,
            files={"none": ""},
            data={
                "prompt": prompt_data["main_prompt"],
                "negative_prompt": prompt_data["negative_prompt"],
                "output_format": "png",
                "cfg_scale": prompt_data["parameters"]["cfg_scale"],
                "steps": prompt_data["parameters"]["steps"],
                "width": 512,
                "height": 768,
            }
        )

        if not response.ok:
            logger.error(f"API Response: {response.text}")
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        # Process response
        img = Image.open(io.BytesIO(response.content))
        logger.info("Successfully generated styled image")

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