"""
Google Gemini API Structured Information Extractor
Communicates with Google AI Studio Gemini API via the google-genai SDK,
enforcing strict JSON schema generation and structured validation.
"""
import json
import re
import time
from typing import Optional, Dict, Any, Tuple
from google import genai
from google.genai import types

from src.extraction.schemas import ExtractedInvoice, LineItem, ExtractionSource
from src.config import settings, get_effective_gemini_api_key, mask_api_key


SYSTEM_INSTRUCTION = """
You are an expert Accounts Payable Auditor and Invoice Data Extraction System for Kaveri Industrial Supplies Pvt Ltd.
Extract the invoice fields from the provided OCR text into strict, valid JSON matching the exact schema below.

Rules:
1. Only extract information directly present or unambiguously indicated in the OCR text.
2. NEVER hallucinate, guess, or invent missing invoice values. If a field is not present or cannot be determined, use null or 0.0.
3. Dates must be formatted as YYYY-MM-DD.
4. Normalize all numbers by removing currency symbols (₹, Rs), commas, and trailing dashes.
5. In India, GSTIN is a 15-character identifier (e.g. 33XXYEN7492X1Z2). Extract the vendor's GSTIN.
6. Return ONLY the JSON object. Do not include markdown codeblocks or conversational filler.

Schema:
{
  "vendor_name": string or null,
  "vendor_gstin": string or null,
  "invoice_number": string or null,
  "invoice_date": "YYYY-MM-DD" or null,
  "due_date": "YYYY-MM-DD" or null,
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
  "subtotal": float,
  "cgst": float,
  "sgst": float,
  "igst": float,
  "grand_total": float,
  "bank_account_last4": string or null
}
"""


class GeminiExtractor:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or get_effective_gemini_api_key()
        self.model_name = model_name or settings.gemini_model
        self._client = None

    def _get_client(self) -> Optional[genai.Client]:
        key = get_effective_gemini_api_key() or self.api_key
        if not key:
            return None
        if self._client is None or getattr(self, "_cached_key", None) != key:
            self._client = genai.Client(api_key=key)
            self._cached_key = key
        return self._client

    def test_connection(self) -> Tuple[bool, str]:
        """Verify API key validity with a minimal test request."""
        client = self._get_client()
        if not client:
            return False, "No Gemini API key configured."
        try:
            resp = client.models.generate_content(
                model=self.model_name,
                contents="ping"
            )
            return True, f"Connected successfully to {self.model_name}!"
        except Exception as e:
            return False, f"Gemini connection failed: {str(e)}"

    def extract(self, ocr_text: str, file_name: Optional[str] = None) -> Tuple[Optional[ExtractedInvoice], Optional[str]]:
        """
        Extract invoice fields using Google Gemini structured output.
        Returns:
            Tuple of (ExtractedInvoice, error_message)
        """
        client = self._get_client()
        if not client:
            return None, "Gemini API key is not configured. Falling back to rule-based extraction."

        prompt = f"Invoice OCR Text to extract:\n---\n{ocr_text}\n---"

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )

                response_text = response.text.strip() if response.text else ""
                # Strip markdown fences if present
                if response_text.startswith("```"):
                    response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
                    response_text = re.sub(r"\s*```$", "", response_text)

                data = json.loads(response_text)

                # Parse line items
                raw_lines = data.get("line_items", []) or []
                lines = []
                for item in raw_lines:
                    if isinstance(item, dict):
                        lines.append(LineItem(
                            description=str(item.get("description", "Item")),
                            hsn=str(item.get("hsn")) if item.get("hsn") else None,
                            quantity=float(item.get("quantity") or 1.0),
                            unit_price=float(item.get("unit_price") or 0.0),
                            amount=float(item.get("amount") or 0.0)
                        ))

                confidences = {
                    "vendor_name": 0.98 if data.get("vendor_name") else 0.0,
                    "vendor_gstin": 0.98 if data.get("vendor_gstin") else 0.0,
                    "invoice_number": 0.98 if data.get("invoice_number") else 0.0,
                    "po_number": 0.98 if data.get("po_number") else 0.0,
                    "invoice_date": 0.98 if data.get("invoice_date") else 0.0,
                    "due_date": 0.98 if data.get("due_date") else 0.0,
                    "grand_total": 0.98 if data.get("grand_total") else 0.0,
                    "bank_account_last4": 0.95 if data.get("bank_account_last4") else 0.0
                }

                sources = {k: ExtractionSource.GEMINI.value for k in confidences.keys()}

                invoice = ExtractedInvoice(
                    file_name=file_name,
                    invoice_number=data.get("invoice_number"),
                    invoice_date=data.get("invoice_date"),
                    due_date=data.get("due_date"),
                    vendor_name=data.get("vendor_name"),
                    vendor_gstin=data.get("vendor_gstin"),
                    po_number=data.get("po_number"),
                    subtotal=float(data.get("subtotal") or 0.0),
                    cgst=float(data.get("cgst") or 0.0),
                    sgst=float(data.get("sgst") or 0.0),
                    igst=float(data.get("igst") or 0.0),
                    grand_total=float(data.get("grand_total") or 0.0),
                    bank_account_last4=str(data.get("bank_account_last4")) if data.get("bank_account_last4") else None,
                    line_items=lines,
                    raw_ocr_text=ocr_text,
                    confidence_scores=confidences,
                    extraction_sources=sources,
                    input_method="upload"
                )
                return invoice, None

            except json.JSONDecodeError as je:
                if attempt == 1:
                    return None, f"Gemini returned invalid JSON: {str(je)}"
                time.sleep(1)
            except Exception as e:
                # Handle rate limits or temporary network glitch
                err_msg = str(e)
                if "503" in err_msg or "rate limit" in err_msg.lower():
                    time.sleep(2)
                    continue
                return None, f"Gemini extraction error: {err_msg}"

        return None, "Failed to extract from Gemini after retries."
