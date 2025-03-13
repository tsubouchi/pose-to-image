import os
import logging
import requests
import json

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def analyze_pose_for_improvements(pose_image_base64: str):
    """
    Analyze pose using Gemini and generate improvement suggestions
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "contents": [{
                "parts":[{
                    "text": """あなたはプロのポーズ指導者です。以下の画像のポーズを分析し、改善点を提案してください。

以下の形式でJSONを返してください:
{
    "pose_analysis": {
        "current_pose": "現在のポーズの詳細な説明",
        "strong_points": ["良い点1", "良い点2"],
        "suggestions": [
            {
                "point": "改善ポイント",
                "suggestion": "具体的な改善方法",
                "reason": "なぜその改善が効果的か"
            }
        ]
    }
}"""
                }, {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": pose_image_base64
                    }
                }]
            }]
        }
        
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
            raise Exception("No JSON content found in response")
            
        json_content = text_response[start:end]
        analysis_result = json.loads(json_content)
        
        return analysis_result["pose_analysis"]
        
    except Exception as e:
        logger.error(f"Error analyzing pose for improvements: {str(e)}")
        return {
            "current_pose": "ポーズの分析中にエラーが発生しました",
            "strong_points": [],
            "suggestions": []
        }
