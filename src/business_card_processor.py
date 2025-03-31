import os
import json
import base64
import hashlib
import pathlib
import requests
from typing import Dict, Any, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BusinessCardProcessor:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_GENAI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=self.api_key)

    def load_image(self, image_source: str) -> Tuple[bytes, str]:
        """Load image from URL or local path and return bytes with hash."""
        try:
            if image_source.startswith(("http://", "https://")):
                print("Loading image from URL...")
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
            else:
                print("Loading image from local path...")
                image_bytes = pathlib.Path(image_source).read_bytes()

            return image_bytes, hashlib.sha256(image_bytes).hexdigest()
        except Exception as e:
            raise RuntimeError(f"Error loading image: {e}")

    def process_llm_response(self, response) -> Dict[str, Any]:
        """Process API response and return extracted data."""
        if not response or not response.text:
            print("No valid response received from the API.")
            return {}

        try:
            cleaned_data = response.text.strip()
            # Handle various JSON response formats
            for wrapper in ["```json", "```"]:
                if cleaned_data.startswith(wrapper) and cleaned_data.endswith("```"):
                    cleaned_data = cleaned_data[len(wrapper) : -3].strip()
            return json.loads(cleaned_data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}\nRaw response: {response.text}")
        except Exception as e:
            print(f"Unexpected error processing response: {e}")
        return {}

    def extract_from_image(self, image_source: str) -> Dict[str, Any]:
        """Extract business card information directly from image."""
        # Load and process image
        image_bytes, image_hash = self.load_image(image_source)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        # Enhanced extraction prompt
        EXTRACTION_PROMPT = """You are an advanced AI business card interpreter with multimodal understanding capabilities. 
Your task is to comprehensively analyze this business card and extract all available information, including both explicit text and implicit context from visual elements.

Key Guidelines:
1. Analyze the ENTIRE card holistically - text, layout, colors, logos, and visual hierarchy
2. For logos: Identify company/organization from logos even if not explicitly stated in text
3. For social media: Extract from icons even if platform names aren't written
4. For contact info: Recognize patterns in formatting (phone, email formats)
5. For addresses: Understand location from maps/coordinates if present
6. For job titles: Infer seniority from position in card hierarchy when unclear

Extract and structure ALL available information in this JSON format:
{
  "primary_info": {
    "name": {"value": "", "confidence": "high/medium/low"},
    "job_title": {"value": "", "confidence": "high/medium/low"},
    "company": {
      "text_value": "",
      "logo_identified": true/false,
      "QRcode_identifies": ture/false
      "confidence": "high/medium/low"
    }
  },
  "contact_info": {
    "emails": [{"value": "", "type": "work/personal", "confidence": ""}],
    "phones": [{"value": "", "type": "work/mobile/fax", "confidence": ""}],
    "addresses": [{"value": "", "type": "work/headquarters", "confidence": ""}]
  },
  "digital_presence": {
    "website": {"value": "", "confidence": ""},
    "social_media": [
      {
        "platform": "linkedin/twitter/etc",
        "handle": "",
        "identified_from": "text/icon",
        "confidence": ""
      }
    ]
  },
  "contextual_summary": {
    "professional_summary": "Detailed professional summery includes name, profession, title, company, location and expertise based on the professional title. Infer relavant working industry and professional filed. Provide seniority estimation and include logical reasoning for the estimation, other than the position or title consider the facts logical like (but not limited to) lack of domain specific email id, personal branding or company employee, website url domain credibility and etc.",
  }
}"""

        # Extract and structure ALL available information in this JSON format:
        # {
        #   "primary_info": {
        #     "name": {"value": "", "confidence": "high/medium/low"},
        #     "job_title": {"value": "", "confidence": "high/medium/low"},
        #     "company": {
        #       "text_value": "",
        #       "logo_identified": true/false,
        #       "confidence": "high/medium/low"
        #     }
        #   },
        #   "contact_info": {
        #     "emails": [{"value": "", "type": "work/personal", "confidence": ""}],
        #     "phones": [{"value": "", "type": "work/mobile/fax", "confidence": ""}],
        #     "addresses": [{"value": "", "type": "work/headquarters", "confidence": ""}]
        #   },
        #   "digital_presence": {
        #     "website": {"value": "", "confidence": ""},
        #     "social_media": [
        #       {
        #         "platform": "linkedin/twitter/etc",
        #         "handle": "",
        #         "identified_from": "text/icon",
        #         "confidence": ""
        #       }
        #     ]
        #   },
        #   "visual_elements": {
        #     "color_scheme": "",
        #     "design_style": "corporate/creative/etc",
        #     "notable_features": ["logo placement", "qr code", etc]
        #   },
        #   "contextual_summary": {
        #     "professional_summary": "Generate a high level comprehensive professional summary. To be used as a context for serach engine to find the indivitual by any relavnt query later.",
        #     "Contact info:"Contact information like all social media info"
        #     "industry_inference": "Infere all relavant fields",
        #     "seniority_estimate": "Should add logical reasoning for the estimation in brackets"
        #   },
        #   "processing_metadata": {
        #     "text_elements_used": [],
        #     "visual_elements_used": [],
        #     "challenges": "low contrast/ambiguous layout/etc"
        #   }
        # }

        # Special Instructions:
        # - Include confidence ratings for all extracted fields
        # - Note whether information came from text or visual interpretation
        # - For ambiguous cases, provide multiple possibilities with confidence levels
        # - Include all visual elements that might convey meaning
        # - Generate a comprehensive professional summary combining all elements"""

        # Make request to Gemini
        print("Extracting business card information with multimodal analysis...")
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[types.Part(text=EXTRACTION_PROMPT), image_part])
            ],
        )

        # Process response and create card data
        extracted_data = self.process_llm_response(response)
        return {
            "image_metadata": {
                "hash": image_hash,
                "base64": image_base64,
            },
            "extracted_info": extracted_data,
        }
