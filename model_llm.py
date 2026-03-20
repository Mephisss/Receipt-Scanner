"""
LLM Vision-based receipt extraction model.
Uses Groq's Llama Vision model for accurate extraction of any receipt format.
Works with Austrian, German, and international receipts.
"""

import re
import json
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional
import os

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class LLMReceiptExtractor:
    """
    Uses a vision-capable LLM to extract receipt data.
    Much better at handling diverse receipt formats than domain-specific models.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Vision model to use (llama-3.2-90b-vision-preview or llama-3.2-11b-vision-preview)
        """
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        logger.info(f"LLMReceiptExtractor initialized with model: {self.model}")

    @staticmethod
    def _get_api_key() -> str:
        """Get API key from environment or config."""
        key = os.environ.get("GROQ_API_KEY")
        if not key:
# env
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("GROQ_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        
        if not key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it as an environment variable or in .env file"
            )
        return key

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
# remove alpha
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    def extract(self, image: Image.Image) -> dict:
        """
        Extract receipt data using vision LLM.
        
        Returns a normalized dict with keys:
            store_name, store_address, date, time,
            items, subtotal, tax, total, savings, payment_method
        """
        image_b64 = self._image_to_base64(image)
        
        prompt = self._build_prompt()
        
# API
        try:
            logger.info("Calling Groq Vision API...")
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            # json
            content = result["choices"][0]["message"]["content"]
            logger.debug(f"Raw LLM response: {content}")
            
            return self._parse_response(content)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"Failed to extract receipt data: {str(e)}")

    def extract_from_path(self, image_path: str) -> dict:
        """Convenience wrapper — accepts a file path string."""
        image = Image.open(image_path)
        return self.extract(image)

    def _build_prompt(self) -> str:
        """Build the extraction prompt."""
        return """Extract ALL information from this receipt image and return it as a JSON object.

CRITICAL INSTRUCTIONS:
1. Read the ENTIRE receipt from top to bottom
2. Extract EVERY SINGLE line item with its price
3. Handle European number format: "1,99" means 1.99 (comma is decimal separator)
4. Extract discount info ("Aktionsersparnis" = discount/savings)
5. Return ONLY valid JSON - no markdown, no code blocks, no extra text

Required JSON format:

{
  "store_name": "exact store name from receipt",
  "store_address": "full address if visible",
  "date": "DD.MM.YYYY format",
  "time": "HH:MM format",
  "items": [
    {
      "name": "clean item name (remove codes/noise)",
      "quantity": 1,
      "unit_price": 0.00,
      "total_price": 0.00,
      "discount": 0.00
    }
  ],
  "subtotal": 0.00,
  "tax": 0.00,
  "total": 0.00,
  "savings": 0.00,
  "payment_method": "VISA/cash/etc",
  "currency": "EUR"
}

Parsing rules:
- Item like "2 × 0,99" means quantity=2, unit_price=0.99, total_price=1.98
- "Aktionsersparnis 0,20" means discount=0.20 for that item
- "Ihre Ersparnis: 2,34 EUR" at bottom = total savings
- Clean item names: "MAMA NOODLES BEEF BT" not "MAMA NOODLES BEEF BT 2 x 0,99"
- Convert ALL comma decimals to dots: 31,67 → 31.67
- Extract EVERY item you see, not just a sample

Return ONLY the JSON object, nothing else."""

    def _parse_response(self, content: str) -> dict:
        """Parse the LLM response and normalize it."""
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()
        

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")
        
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Content: {content[:500]}")
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        
# normalized
        return {
            "store_name": data.get("store_name", ""),
            "store_address": data.get("store_address", ""),
            "date": data.get("date", ""),
            "time": data.get("time", ""),
            "items": self._normalize_items(data.get("items", [])),
            "subtotal": self._to_float(data.get("subtotal", 0)),
            "tax": self._to_float(data.get("tax", 0)),
            "total": self._to_float(data.get("total", 0)),
            "savings": self._to_float(data.get("savings", 0)),
            "payment_method": data.get("payment_method", ""),
            "currency": data.get("currency", "EUR"),
            "confidence": self._estimate_confidence(data),
            "raw_json": data
        }

    def _normalize_items(self, items: list) -> list:
        """Normalize item structure."""
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            
            normalized.append({
                "name": str(item.get("name", "Unknown")).strip(),
                "quantity": self._to_int(item.get("quantity", 1)),
                "unit_price": self._to_float(item.get("unit_price", 0)),
                "total_price": self._to_float(item.get("total_price", 0)),
                "discount": self._to_float(item.get("discount", 0))
            })
        
        return normalized

# cleanup
    @staticmethod
    def _to_int(value) -> int:
        """Convert various formats to int."""
        if value is None:
            return 1
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        

        cleaned = str(value).strip()
        cleaned = re.sub(r'[^\d]', '', cleaned)
        
        try:
            return int(cleaned) if cleaned else 1
        except ValueError:
            return 1

    @staticmethod
    def _to_float(value) -> float:
        """Convert various number formats to float."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)

        cleaned = str(value).replace(",", ".").strip()
  
        cleaned = re.sub(r"[^\d.-]", "", cleaned)
        
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    @staticmethod
    def _estimate_confidence(data: dict) -> float:
        """Estimate extraction confidence based on completeness."""
        score = 0.0
        

        if data.get("store_name"):
            score += 20.0

        if data.get("date"):
            score += 10.0
        

        items = data.get("items", [])
        if items:
            score += 30.0
            valid_items = sum(1 for item in items if item.get("total_price", 0) > 0)
            score += min(valid_items * 5, 20.0)
        
        if data.get("total", 0) > 0:
            score += 20.0
        
        return min(score, 100.0)