from __future__ import annotations

import io
import os
import re
from collections.abc import Iterable

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


ALCOHOL_PATTERNS = re.compile(
    r"\b(beer|wine|vodka|whiskey|whisky|rum|gin|tequila|champagne|cider|lager)\b",
    flags=re.IGNORECASE,
)


def _normalize_amount(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    # Keep only numeric characters/signs typically seen in currency strings.
    cleaned = re.sub(r"[^\d.,-]", "", value).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _flatten_text_values(data: object) -> Iterable[str]:
    # Recursively flatten nested OCR JSON into text so fallback parsing can scan all content.
    if isinstance(data, str):
        yield data
    elif isinstance(data, list):
        for item in data:
            yield from _flatten_text_values(item)
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str):
                yield key
            yield from _flatten_text_values(value)


class DonutReceiptExtractor:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "DONUT_MODEL", "naver-clova-ix/donut-base-finetuned-cord-v2"
        )
        # Processor handles image preprocessing, prompt tokenization, and decoded output parsing.
        self.processor = DonutProcessor.from_pretrained(self.model_name, use_fast = True)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert image into model-ready tensors.
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(
            self.device
        )
        # CORD task token steers generation toward receipt-style structured output.
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Generate structured sequence and constrain unknown token generation.
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0]
        # Donut utility converts generated markup-like text into a JSON-like dict.
        parsed = self.processor.token2json(sequence)
        return self._map_to_fields(parsed)

    def _map_to_fields(self, parsed: dict) -> dict:
        # Flattened text is used only as a fallback when line items are not explicitly present.
        text_dump = " ".join(_flatten_text_values(parsed))

        merchant_name = None
        # Probe common key variants since different checkpoints may emit different field names.
        for key in ("store", "merchant", "company", "seller"):
            if key in parsed:
                merchant_name = parsed[key]
                break

        date_value = None
        for key in ("date", "transaction_date", "purchase_date"):
            if key in parsed:
                date_value = parsed[key]
                break

        total_amount = None
        currency = None
        total_block = parsed.get("total")
        if isinstance(total_block, dict):
            # Total can be nested with varying names depending on dataset/checkpoint.
            total_amount = _normalize_amount(
                total_block.get("price") or total_block.get("value")
            )
            currency = total_block.get("currency")
        if total_amount is None and "amount" in parsed:
            total_amount = _normalize_amount(parsed.get("amount"))

        line_items: list[str] = []
        # Collect item names from multiple likely keys and item structures.
        for key in ("menu", "items", "line_items"):
            items = parsed.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        name = item.get("nm") or item.get("name") or item.get("item")
                        if name:
                            line_items.append(str(name))
                    elif isinstance(item, str):
                        line_items.append(item)

        if not line_items:
            # Fallback: pull plausible line-like phrases from flattened text dump.
            possible_lines = re.findall(r"[A-Za-z][A-Za-z0-9\s&\-]{2,}", text_dump)
            line_items = possible_lines[:25]

        # Simple policy flag derived from extracted line items.
        has_alcohol_item = any(
            ALCOHOL_PATTERNS.search(item or "") for item in line_items
        )

        return {
            "merchant_name": merchant_name if isinstance(merchant_name, str) else None,
            "date": date_value if isinstance(date_value, str) else None,
            "amount": total_amount,
            "currency": currency if isinstance(currency, str) else None,
            "line_items": line_items,
            "has_alcohol_item": bool(has_alcohol_item),
            "raw": parsed,
        }


class LayoutLMv3ReceiptExtractor:
    def extract(self, image_bytes: bytes) -> dict:
        # LayoutLMv3 requires a fine-tuned token-classification checkpoint and OCR boxes.
        # This placeholder keeps the strategy plug-compatible with Donut.
        raise NotImplementedError(
            "LayoutLMv3 extractor is not wired. Provide a fine-tuned checkpoint and OCR box pipeline."
        )


def build_ocr_extractor() -> object:
    engine = os.getenv("OCR_ENGINE", "donut").lower().strip()
    # Keep OCR backend pluggable through environment configuration.
    if engine == "donut":
        return DonutReceiptExtractor()
    if engine == "layoutlmv3":
        return LayoutLMv3ReceiptExtractor()
    raise ValueError(f"Unsupported OCR engine: {engine}")
