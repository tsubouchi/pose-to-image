import os
import base64
import logging
import tempfile
from PIL import Image
import requests
import io
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

def parse_gemini_response(response_text: str) -> dict:
    """
    Parse Gemini API response text to extract JSON content
    """
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            logger.error("No JSON content found in response")
            logger.error(f"Full response: {response_text}")
            raise Exception("No JSON content found in response")

        json_content = response_text[start:end]
        json_content = json_content.replace("```json", "").replace("```", "").strip()

        result = json.loads(json_content)

        if "main_prompt" not in result or "negative_prompt" not in result:
            raise Exception("Missing required fields in response")

        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Attempted to parse: {json_content}")
        return {
            "main_prompt": "masterpiece, best quality, highly detailed, maintain exact pose",
            "negative_prompt": "low quality, blurry, distorted",
            "parameters": {"cfg_scale": 7, "steps": 20}
        }
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {str(e)}")
        logger.error(f"Response text: {response_text}")
        raise Exception(f"Failed to parse Gemini response: {str(e)}")

def analyze_images_with_llm(pose_image: Image.Image, style_image: Image.Image):
    """
    Use Gemini to analyze both images and provide detailed descriptions
    """
    try:
        # Convert images to base64
        pose_bytes = io.BytesIO()
        style_bytes = io.BytesIO()
        pose_image.save(pose_bytes, format='PNG')
        style_image.save(style_bytes, format='PNG')

        pose_base64 = base64.b64encode(pose_bytes.getvalue()).decode('utf-8')
        style_base64 = base64.b64encode(style_bytes.getvalue()).decode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"

        headers = {
            'Content-Type': 'application/json'
        }

        data = {
            "contents": [{
                "parts":[{
                    "text": """Please analyze these two images:

FIRST IMAGE - POSE ONLY:
Focus exclusively on body positioning and pose, ignore style and clothing.
Describe:
- Exact body position and orientation
- Specific pose details and gestures
- Key pose points and angles

SECOND IMAGE - STYLE AND CLOTHING:
Analyze complete visual style and clothing details.
Describe:
1. Art Style:
   - Overall artistic style (anime, realistic, etc.)
   - Art technique and rendering approach
   - Visual effects and treatment

2. Clothing Details:
   - All garment pieces and types
   - Colors and patterns
   - Materials and textures
   - Accessories and decorations

3. Visual Elements:
   - Lighting and shading
   - Color scheme and mood
   - Background treatment

Format response EXACTLY as follows:
{
  "pose_reference": {
    "body_position": "detailed position description",
    "gestures": ["specific pose elements"],
    "key_points": ["important angles and positions"]
  },
  "style_reference": {
    "art_style": {
      "type": "main style (anime/realistic/etc)",
      "technique": "artistic approach",
      "effects": ["visual effects"]
    },
    "clothing": {
      "garments": ["all clothing items"],
      "colors": ["color details"],
      "materials": ["fabric and texture"],
      "accessories": ["decorative elements"]
    },
    "visuals": {
      "lighting": "lighting description",
      "color_scheme": "overall color treatment",
      "background": "background style"
    }
  }
}"""
                }, {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": pose_base64
                    }
                }, {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": style_base64
                    }
                }]
            }]
        }

        logger.debug("Sending request to Gemini API")
        response = requests.post(url, headers=headers, json=data)

        if not response.ok:
            logger.error(f"Gemini API Response: {response.text}")
            raise Exception(f"Gemini API error: {response.status_code}")

        result = response.json()
        if not result.get("candidates"):
            raise Exception("No candidates in Gemini response")

        text_response = result["candidates"][0]["content"]["parts"][0]["text"]

        # Extract JSON content
        start = text_response.find('{')
        end = text_response.rfind('}') + 1

        if start == -1 or end == 0:
            logger.error(f"No JSON found in response: {text_response}")
            raise Exception("Failed to extract JSON from response")

        json_str = text_response[start:end]
        return json.loads(json_str)

    except Exception as e:
        logger.error(f"Error in analyze_images_with_llm: {str(e)}")
        logger.error(f"Full error context: {str(e.__class__.__name__)}")
        return None

def generate_enhanced_prompt(analysis):
    """
    Generate a detailed prompt based on the analysis
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"

        headers = {
            'Content-Type': 'application/json'
        }

        data = {
            "contents": [{
                "parts":[{
                    "text": f"""Generate a Stable Diffusion prompt to recreate the EXACT pose from the first image while applying the COMPLETE style from the second image.

Analysis result:
{json.dumps(analysis, indent=2)}

Create a prompt in this format:
{{
  "main_prompt": "masterpiece, best quality, highly detailed, (art style from reference), [clothing details], [pose description], [visual elements]",
  "negative_prompt": "wrong pose, wrong style, low quality, blurry, distorted",
  "parameters": {{
    "cfg_scale": 7,
    "steps": 20
  }}
}}

Requirements:
1. Art style: Use EXACTLY the style from the reference image
2. Clothing: Include ALL clothing details from reference
3. Pose: Keep EXACT pose from first image
4. Add style elements from reference: lighting, effects, mood

Return ONLY the JSON object, no additional text."""
                }]
            }]
        }

        logger.debug("Sending prompt generation request to Gemini")
        response = requests.post(url, headers=headers, json=data)

        if not response.ok:
            logger.error(f"Gemini API Response: {response.text}")
            raise Exception(f"Gemini API error: {response.status_code}")

        result = response.json()
        if not result.get("candidates"):
            raise Exception("No candidates in Gemini response")

        text_response = result["candidates"][0]["content"]["parts"][0]["text"]

        # Clean and parse JSON response
        start = text_response.find('{')
        end = text_response.rfind('}') + 1

        if start == -1 or end == 0:
            raise Exception("No JSON content found in response")

        json_content = text_response[start:end].strip()
        prompt_data = json.loads(json_content)

        # Validate required fields
        if not all(key in prompt_data for key in ["main_prompt", "negative_prompt", "parameters"]):
            raise Exception("Missing required fields in prompt data")

        return prompt_data

    except Exception as e:
        logger.error(f"Error generating enhanced prompt: {str(e)}")
        # Return default prompt as fallback
        return {
            "main_prompt": "masterpiece, best quality, highly detailed, maintain exact pose",
            "negative_prompt": "wrong pose, low quality, blurry, distorted",
            "parameters": {"cfg_scale": 7, "steps": 20}
        }

def generate_image_with_style(pose_image, style_image):
    """
    Generate a new image that combines the pose from pose_image with the style from style_image
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