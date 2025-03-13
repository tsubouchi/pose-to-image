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
        # Find the JSON content between curly braces
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            logger.error("No JSON content found in response")
            logger.error(f"Full response: {response_text}")
            raise Exception("No JSON content found in response")

        json_content = response_text[start:end]

        # Remove any markdown code block markers
        json_content = json_content.replace("```json", "").replace("```", "").strip()

        # Parse and validate JSON
        result = json.loads(json_content)

        # Validate required fields
        if "main_prompt" not in result or "negative_prompt" not in result:
            raise Exception("Missing required fields in response")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Attempted to parse: {json_content}")
        # Return default prompt as fallback
        return {
            "main_prompt": "masterpiece, best quality, highly detailed, full body pose with exact matching",
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
                    "text": """Analyze these two images for a style transfer task:

1. First Image (Pose Reference Only):
- Describe only the pose and body positioning
- Focus on body language and gesture
- Do not include any style or clothing details from this image

2. Second Image (Style and Clothing Reference):
- Detailed clothing analysis:
  * All garment types and pieces
  * Specific design elements
  * Materials and textures
  * Fit and cut details
  * Colors and patterns
  * Accessories
- Style elements:
  * Overall artistic style
  * Visual effects
  * Lighting and atmosphere
  * Color scheme and mood
  * Composition

Format the response exactly like this:
{
  "pose": {
    "body_position": "detailed pose description",
    "gesture": "body language details",
    "key_points": ["important pose elements"]
  },
  "reference_style": {
    "clothing": {
      "garments": ["detailed list of clothing items"],
      "design": ["specific design elements"],
      "materials": ["fabric and texture details"],
      "colors": ["color palette"],
      "accessories": ["all accessories"]
    },
    "artistic": {
      "style": "overall artistic style",
      "lighting": "lighting description",
      "atmosphere": "mood and ambiance",
      "effects": ["visual effects"]
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

        # Find the JSON content between curly braces
        start = text_response.find('{')
        end = text_response.rfind('}') + 1
        if start == -1 or end == 0:
            raise Exception("No JSON content found in response")

        json_content = text_response[start:end]
        return json.loads(json_content)

    except Exception as e:
        logger.error(f"Error in Gemini analysis: {str(e)}")
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
                    "text": f"""Create a detailed Stable Diffusion prompt that recreates the exact pose from the first image 
while applying the style and clothing from the second image.

Analysis result:
{json.dumps(analysis, indent=2)}

Requirements:
1. Quality and style tags:
   - Start with: masterpiece, best quality, highly detailed
   - Include reference style's artistic approach
2. Clothing description (from reference style):
   - Exact garment details
   - Materials and textures
   - Colors and patterns
   - Accessories
3. Pose description (from pose reference):
   - Precise body position
   - Gesture and attitude
4. Visual style (from reference style):
   - Lighting and atmosphere
   - Color grading
   - Special effects
5. Technical aspects:
   - Camera angle
   - Composition
   - Background elements

Format the response EXACTLY like this:
{{
  "main_prompt": "masterpiece, best quality, [clothing], [pose], [style], [effects]",
  "negative_prompt": "wrong clothes, wrong style, poor quality, blurry, distorted",
  "parameters": {{
    "cfg_scale": 7,
    "steps": 20
  }}
}}"""
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

        # Find the JSON content between curly braces
        start = text_response.find('{')
        end = text_response.rfind('}') + 1
        if start == -1 or end == 0:
            raise Exception("No JSON content found in response")

        json_content = text_response[start:end]
        return json.loads(json_content)

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