"""Gemini LLM-based Invoice Information Extraction."""

import json
import re
from typing import Dict, Any, Optional
from src.extraction.schemas import InvoiceExtraction, LineItem
from config.settings import settings
from config.logging_config import logger

GEMINI_PROMPT_TEMPLATE = """You are an expert accounts-payable invoice extraction system.
Extract structured invoice data from the following OCR text.

Strict requirements:
1. Return ONLY a valid JSON object matching the schema below.
2. DO NOT invent, guess, or hallucinate missing values. If a field is not present or cannot be read, set it to null.
3. Normalize Indian currency numbers to float (e.g. "1,23,456.00" -> 123456.00).
4. Normalize dates to standard ISO "YYYY-MM-DD" format.
5. Extract all line items with description, hsn, quantity, unit_price, and amount.

JSON Schema:
{
  "vendor_name": string or null,
  "vendor_gstin": string (15 chars) or null,
  "invoice_number": string or null,
  "invoice_date": string (YYYY-MM-DD) or null,
  "due_date": string (YYYY-MM-DD) or null,
  "po_number": string or null,
  "line_items": [
    {
      "description": string,
      "hsn": string or null,
      "quantity": float,
      "unit_price": float,
      "amount": float
    }
  ],
  "subtotal": float or null,
  "cgst": float or 0.0,
  "sgst": float or 0.0,
  "igst": float or 0.0,
  "grand_total": float or null,
  "bank_account_last4": string (4 digits) or null
}

OCR Text:
---
{ocr_text}
---
"""

class GeminiExtractor:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self._client = None

    def _get_client(self):
        if not self.api_key:
            return None
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to create Google GenAI client: {e}")
                self._client = None
        return self._client

    def is_configured(self) -> bool:
        """Check whether an API key is configured."""
        return bool(self.api_key and len(self.api_key) > 10)

    def test_connection(self) -> Dict[str, Any]:
        """Test API key connection with a minimal prompt."""
        if not self.is_configured():
            return {"success": False, "message": "Gemini API key is not configured."}
        
        client = self._get_client()
        if not client:
            return {"success": False, "message": "Failed to initialize Google GenAI SDK client."}
            
        try:
            # Try configured model, fallback if needed
            for test_model in [self.model, "gemini-3.5-flash-lite", "gemini-flash-latest"]:
                try:
                    response = client.models.generate_content(
                        model=test_model,
                        contents="Respond with: OK",
                    )
                    if response and response.text:
                        return {
                            "success": True,
                            "model": test_model,
                            "message": f"Connection verified successfully using {test_model}!"
                        }
                except Exception as model_err:
                    err_msg = str(model_err)
                    if "404" in err_msg or "not found" in err_msg.lower():
                        continue
                    raise model_err
            return {"success": False, "message": "No active Gemini model responded successfully."}
        except Exception as e:
            logger.error(f"Gemini test connection failed: {e}")
            return {"success": False, "message": f"API Error: {str(e)}"}

    def extract(self, ocr_result: Dict[str, Any]) -> Optional[InvoiceExtraction]:
        """Extract structured invoice data using Gemini."""
        if not self.is_configured():
            logger.info("Gemini API key not configured. Skipping LLM extraction.")
            return None

        client = self._get_client()
        if not client:
            return None

        ocr_text = ocr_result.get("raw_text", "")
        if not ocr_text.strip():
            logger.warning("Empty OCR text provided to Gemini.")
            return None

        try:
            prompt = GEMINI_PROMPT_TEMPLATE.replace("{ocr_text}", ocr_text)

            from google.genai import types
            response = None
            models_to_try = [self.model]
            if "gemini-3.5-flash-lite" not in models_to_try:
                models_to_try.append("gemini-3.5-flash-lite")

            for m in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    if response and response.text:
                        break
                except Exception as m_err:
                    logger.warning(f"Gemini model {m} extraction error: {m_err}. Trying fallback model.")
                    continue

            if not response or not response.text:
                logger.warning("No Gemini response received. Falling back to rule extraction.")
                return None

            text_resp = response.text.strip()
            # Clean markdown codeblocks if wrapped
            if text_resp.startswith("```"):
                text_resp = re.sub(r"^```(?:json)?\s*", "", text_resp)
                text_resp = re.sub(r"\s*```$", "", text_resp)

            data = json.loads(text_resp)

            # Build Line Items
            line_items = []
            for item in data.get("line_items", []):
                line_items.append(LineItem(
                    description=item.get("description", "Supplies"),
                    hsn=str(item.get("hsn")) if item.get("hsn") else None,
                    quantity=float(item.get("quantity") or 1.0),
                    unit_price=float(item.get("unit_price") or 0.0),
                    amount=float(item.get("amount") or 0.0)
                ))

            # Build confidence scores
            confidences = {
                "vendor_name": 0.95 if data.get("vendor_name") else 0.0,
                "vendor_gstin": 0.95 if data.get("vendor_gstin") else 0.0,
                "invoice_number": 0.95 if data.get("invoice_number") else 0.0,
                "invoice_date": 0.95 if data.get("invoice_date") else 0.0,
                "due_date": 0.90 if data.get("due_date") else 0.0,
                "po_number": 0.95 if data.get("po_number") else 0.0,
                "grand_total": 0.95 if data.get("grand_total") else 0.0,
                "subtotal": 0.95 if data.get("subtotal") else 0.0,
                "bank_account_last4": 0.90 if data.get("bank_account_last4") else 0.0
            }

            return InvoiceExtraction(
                vendor_name=data.get("vendor_name"),
                vendor_gstin=data.get("vendor_gstin"),
                invoice_number=data.get("invoice_number"),
                invoice_date=data.get("invoice_date"),
                due_date=data.get("due_date"),
                po_number=data.get("po_number"),
                line_items=line_items,
                subtotal=float(data.get("subtotal")) if data.get("subtotal") is not None else None,
                cgst=float(data.get("cgst") or 0.0),
                sgst=float(data.get("sgst") or 0.0),
                igst=float(data.get("igst") or 0.0),
                grand_total=float(data.get("grand_total")) if data.get("grand_total") is not None else None,
                bank_account_last4=str(data.get("bank_account_last4")) if data.get("bank_account_last4") else None,
                extraction_source="gemini",
                confidence_scores=confidences,
                raw_ocr_text=ocr_text
            )

        except Exception as e:
            logger.warning(f"Gemini extraction failed: {e}. Falling back to rule extraction.")
            return None
