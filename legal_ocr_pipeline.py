#!/usr/bin/env python3
"""
Production OCR pipeline for bilingual legal documents (English + Arabic).

Features
--------
- Accepts PDF and common image formats.
- Uses Docling for PDF text extraction when available.
- Uses PaddleOCR for OCR on scanned PDFs and images.
- Parses legal hierarchy into structured JSON:
  Document -> Parts -> Chapters -> Articles
- Supports mixed structures:
  - document-level articles
  - parts with direct articles
  - parts with chapters and articles
  - top-level chapters such as "Final Provisions"
- Splits title/article content into English/Arabic fields.

Example
-------
python3 legal_ocr_pipeline.py input.pdf -o output.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("legal_ocr_pipeline")


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"
}
PDF_RENDER_DPI = 200
MAX_IMAGE_SIDE = 1600
ARABIC_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
WHITESPACE_RE = re.compile(r"\s+")

PART_RE = re.compile(
    r"^\s*(?:(?:Part|PART)\b|الجزء|الباب)\s*[\(\[]?\s*([A-Za-z0-9٠-٩IVXLC]+)\s*[\)\]]?\s*[:.\-–—]?\s*(.*)\s*$",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(
    r"^\s*(?:(?:Chapter|CHAPTER)\b|الفصل)\s*[\(\[]?\s*([A-Za-z0-9٠-٩IVXLC]+)\s*[\)\]]?\s*[:.\-–—]?\s*(.*)\s*$",
    re.IGNORECASE,
)
ARTICLE_WITH_BODY_RE = re.compile(
    r"^\s*(?:(?:Article|ARTICLE)\b|المادة|مادة)\s*[\(\[]?\s*([0-9٠-٩]+)\s*[\)\]]?\s*[:\-–—]\s*(.*)\s*$",
    re.IGNORECASE,
)
ARTICLE_STANDALONE_RE = re.compile(
    r"^\s*(?:(?:Article|ARTICLE)\b|المادة|مادة)\s*[\(\[]?\s*([0-9٠-٩]+)\s*[\)\]]?\s*:?\s*$",
    re.IGNORECASE,
)
FINAL_PROVISIONS_RE = re.compile(
    r"^\s*(?:Final\s+Provisions?|General\s+Provisions?|Concluding\s+Provisions?|الأحكام\s+الختامية|احكام\s+ختامية)\s*$",
    re.IGNORECASE,
)

ARABIC_NUMBER_WORDS = {
    "الاول": 1,
    "الأول": 1,
    "الاولى": 1,
    "الأولى": 1,
    "الثاني": 2,
    "الثانية": 2,
    "الثالث": 3,
    "الثالثة": 3,
    "الرابع": 4,
    "الرابعة": 4,
    "الخامس": 5,
    "الخامسة": 5,
    "السادس": 6,
    "السادسة": 6,
    "السابع": 7,
    "السابعة": 7,
    "الثامن": 8,
    "الثامنة": 8,
    "التاسع": 9,
    "التاسعة": 9,
    "العاشر": 10,
    "العاشرة": 10,
    "الحادي عشر": 11,
    "الحادية عشرة": 11,
    "الثاني عشر": 12,
    "الثانية عشرة": 12,
    "الثالث عشر": 13,
    "الثالثة عشرة": 13,
    "الرابع عشر": 14,
    "الرابعة عشرة": 14,
    "الخامس عشر": 15,
    "الخامسة عشرة": 15,
    "السادس عشر": 16,
    "السادسة عشرة": 16,
    "السابع عشر": 17,
    "السابعة عشرة": 17,
    "الثامن عشر": 18,
    "الثامنة عشرة": 18,
    "التاسع عشر": 19,
    "التاسعة عشرة": 19,
    "العشرون": 20,
    "العشرين": 20,
    "الثلاثون": 30,
    "الثلاثين": 30,
    "الثلائون": 30,
    "الثلائين": 30,
    "الاربعون": 40,
    "الأربعون": 40,
    "الاربعين": 40,
    "الأربعين": 40,
    "الخمسون": 50,
    "الخمسين": 50,
    "الستون": 60,
    "الستين": 60,
}


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. Install with: {install_hint}") from exc


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u200f", " ").replace("\u200e", " ")
    return WHITESPACE_RE.sub(" ", value).strip()


def normalize_multiline_text(value: str) -> str:
    lines = value.replace("\r", "\n").split("\n")
    return "\n".join(normalize_text(line) for line in lines if normalize_text(line))


def normalize_digits(value: str) -> str:
    return value.translate(ARABIC_DIGIT_TRANSLATION)


def normalize_arabic_wording(value: str) -> str:
    value = normalize_text(value)
    value = value.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    return value


def parse_arabic_number_words(value: str) -> Optional[int]:
    text = normalize_arabic_wording(value)
    if text in ARABIC_NUMBER_WORDS:
        return ARABIC_NUMBER_WORDS[text]

    parts = [part.strip() for part in re.split(r"\s+و", text) if part.strip()]
    if len(parts) == 2:
        left = ARABIC_NUMBER_WORDS.get(parts[0])
        right = ARABIC_NUMBER_WORDS.get(parts[1])
        if left is not None and right is not None:
            return left + right

    return None


def extract_leading_arabic_number_phrase(value: str) -> Optional[tuple[str, str]]:
    tokens = normalize_text(value).split()
    max_size = min(4, len(tokens))
    for size in range(max_size, 0, -1):
        candidate = " ".join(tokens[:size])
        parsed_number = parse_arabic_number_words(candidate)
        if parsed_number is not None:
            remainder = normalize_text(" ".join(tokens[size:]))
            return str(parsed_number), remainder
    return None


def contains_arabic(value: str) -> bool:
    return bool(ARABIC_CHAR_RE.search(value or ""))


def contains_latin(value: str) -> bool:
    return bool(LATIN_CHAR_RE.search(value or ""))


def clean_script_projection(value: str, mode: str) -> Optional[str]:
    pieces: list[str] = []
    for token in normalize_text(value).split():
        has_ar = contains_arabic(token)
        has_en = contains_latin(token)
        if mode == "ar" and has_ar:
            pieces.append(token)
        elif mode == "en" and has_en:
            pieces.append(token)
        elif mode == "ar" and not has_en and not has_ar and pieces:
            pieces.append(token)
        elif mode == "en" and not has_en and not has_ar and pieces:
            pieces.append(token)
    projected = normalize_text(" ".join(pieces))
    return projected or None


def split_bilingual_text(value: str) -> tuple[Optional[str], Optional[str]]:
    text = normalize_multiline_text(value)
    if not text:
        return None, None
    has_ar = contains_arabic(text)
    has_en = contains_latin(text)
    if has_en and not has_ar:
        return text, None
    if has_ar and not has_en:
        return None, text
    if not has_ar and not has_en:
        return text, None
    text_en = clean_script_projection(text, "en")
    text_ar = clean_script_projection(text, "ar")
    if not text_en and has_en:
        text_en = text
    if not text_ar and has_ar:
        text_ar = text
    return text_en, text_ar


def is_list_item_line(value: str) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    return bool(
        re.match(
            r"^(?:\d+[\.\)]|[٠-٩]+[\.\)]|[A-Za-z][\.\)]|[اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويى][\.\)]?)\s+",
            text,
        )
    )


def format_article_text(lines: list[str]) -> str:
    normalized_lines = [normalize_text(raw_line) for raw_line in lines]
    normalized_lines = [line for line in normalized_lines if line]
    return "\n".join(normalized_lines).strip()


@dataclass
class OCRLine:
    text: str
    confidence: float
    page_number: int
    x: int
    y: int
    width: int
    height: int


@dataclass
class ArticleContext:
    article_number: str
    order_index: int
    lines: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        text = format_article_text(self.lines)
        text_en, text_ar = split_bilingual_text(text)
        return {
            "article_number": self.article_number,
            "text_en": text_en,
            "text_ar": text_ar,
            "order_index": self.order_index,
        }


def build_document_template() -> dict[str, Any]:
    return {
        "title_en": None,
        "title_ar": None,
        "document_type": "law",
        "articles": [],
        "chapters": [],
        "parts": [],
    }


def build_part(title: str, number: str) -> dict[str, Any]:
    title_en, title_ar = split_bilingual_text(title)
    return {
        "title_en": title_en,
        "title_ar": title_ar,
        "part_number": normalize_digits(number),
        "articles": [],
        "chapters": [],
    }


def build_chapter(title: str, number: Optional[str]) -> dict[str, Any]:
    title_en, title_ar = split_bilingual_text(title)
    return {
        "title_en": title_en,
        "title_ar": title_ar,
        "chapter_number": normalize_digits(number) if number is not None else None,
        "articles": [],
    }


def extract_title_from_heading(keyword: str, number: Optional[str], remainder: str) -> str:
    remainder = normalize_text(remainder)
    if remainder:
        if number is None:
            return remainder
        return f"{keyword} {normalize_digits(number)}: {remainder}"
    if number is None:
        return keyword
    return f"{keyword} {normalize_digits(number)}"


def iter_input_pages(input_path: Path) -> Iterator[Any]:
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        pdf2image = require_dependency("pdf2image", "pip install pdf2image")
        from PIL import Image  # type: ignore

        info = pdf2image.pdfinfo_from_path(str(input_path))
        total_pages = int(info["Pages"])
        for page_number in range(1, total_pages + 1):
            page_images = pdf2image.convert_from_path(
                str(input_path),
                dpi=PDF_RENDER_DPI,
                first_page=page_number,
                last_page=page_number,
            )
            if not page_images:
                continue
            try:
                yield page_images[0].convert("RGB")
            finally:
                for page_image in page_images:
                    page_image.close()
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        from PIL import Image  # type: ignore

        image = Image.open(str(input_path))
        try:
            yield image.convert("RGB")
        finally:
            image.close()
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported formats: PDF, {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )


def preprocess_image(pil_image: Any) -> Any:
    cv2 = require_dependency("cv2", "pip install opencv-python-headless")
    np = require_dependency("numpy", "pip install numpy")

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest_side
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    elif longest_side < 1400:
        scale = 1400 / longest_side
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=15)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _iter_prediction_lines(prediction: Any) -> Iterable[Any]:
    if prediction is None:
        return []
    if isinstance(prediction, list):
        if len(prediction) == 1 and isinstance(prediction[0], list):
            return prediction[0]
        return prediction

    if isinstance(prediction, dict) or hasattr(prediction, "__getitem__"):
        try:
            polys = prediction["rec_polys"] or prediction["dt_polys"] or []
        except Exception:
            try:
                polys = prediction["dt_polys"] or []
            except Exception:
                polys = []
        try:
            texts = prediction["rec_texts"] or []
        except Exception:
            texts = []
        try:
            scores = prediction["rec_scores"] or []
        except Exception:
            scores = []
        if polys or texts or scores:
            return list(zip(polys, texts, scores))

    polys = getattr(prediction, "rec_polys", None) or getattr(prediction, "dt_polys", None) or []
    texts = getattr(prediction, "rec_texts", None) or []
    scores = getattr(prediction, "rec_scores", None) or []
    return list(zip(polys, texts, scores))


def init_paddle_engine(lang: str) -> Any:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    paddleocr_module = require_dependency("paddleocr", "pip install paddleocr")
    return paddleocr_module.PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=960,
        text_recognition_batch_size=1,
        textline_orientation_batch_size=1,
        lang=lang,
    )


def run_paddle_ocr(input_path: Path, language_mode: str = "auto") -> list[OCRLine]:
    if language_mode not in {"auto", "eng", "ara"}:
        raise ValueError("language_mode must be one of: auto, eng, ara")

    engine_specs = [("en", "eng"), ("ar", "ara")] if language_mode == "auto" else [
        ("en", "eng") if language_mode == "eng" else ("ar", "ara")
    ]

    engines: list[tuple[str, Any]] = []
    for paddle_lang, _ in engine_specs:
        logger.info("Initializing PaddleOCR model for '%s'", paddle_lang)
        engines.append((paddle_lang, init_paddle_engine(paddle_lang)))

    merged_lines: list[OCRLine] = []
    for page_number, page in enumerate(iter_input_pages(input_path), start=1):
        logger.info("OCR page %s", page_number)
        page_start = time.time()
        processed = preprocess_image(page)
        logger.info("Page %s preprocessed in %.2fs", page_number, time.time() - page_start)
        page_lines: list[OCRLine] = []

        for engine_lang, engine in engines:
            engine_start = time.time()
            logger.info("Running PaddleOCR '%s' on page %s", engine_lang, page_number)
            predictions = list(engine.predict(processed))
            logger.info(
                "PaddleOCR '%s' finished page %s in %.2fs",
                engine_lang,
                page_number,
                time.time() - engine_start,
            )
            for prediction in predictions:
                for entry in _iter_prediction_lines(prediction):
                    if not entry:
                        continue
                    if len(entry) == 2:
                        bbox, text_info = entry
                        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                            continue
                        text, confidence = text_info[0], text_info[1]
                    elif len(entry) == 3:
                        bbox, text, confidence = entry
                    else:
                        continue

                    text = normalize_text(str(text))
                    if not text:
                        continue

                    try:
                        xs = [int(float(point[0])) for point in bbox]
                        ys = [int(float(point[1])) for point in bbox]
                    except Exception:
                        continue

                    page_lines.append(
                        OCRLine(
                            text=text,
                            confidence=float(confidence),
                            page_number=page_number,
                            x=min(xs),
                            y=min(ys),
                            width=max(xs) - min(xs),
                            height=max(ys) - min(ys),
                        )
                    )

        merged_lines.extend(deduplicate_lines(page_lines))
        logger.info(
            "Page %s produced %s OCR lines in %.2fs",
            page_number,
            len(page_lines),
            time.time() - page_start,
        )
        page.close()
    return merged_lines


def deduplicate_lines(lines: list[OCRLine]) -> list[OCRLine]:
    def sort_key(item: OCRLine) -> tuple[int, int, int]:
        return item.page_number, item.y, item.x

    deduped: list[OCRLine] = []
    seen: set[tuple[int, int, str]] = set()
    for line in sorted(lines, key=sort_key):
        key = (line.page_number, round(line.y / 12), normalize_text(line.text).casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


def lines_to_text(lines: list[OCRLine]) -> str:
    ordered = sorted(lines, key=lambda item: (item.page_number, item.y, item.x))
    return "\n".join(line.text for line in ordered if normalize_text(line.text))


def extract_text_with_docling(input_path: Path, language_mode: str = "auto") -> Optional[str]:
    if input_path.suffix.lower() != ".pdf":
        return None
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        return None

    try:
        logger.info("Attempting Docling extraction")
        if language_mode == "eng":
            tesseract_langs = ["eng"]
        elif language_mode == "ara":
            tesseract_langs = ["ara"]
        else:
            tesseract_langs = ["eng", "ara"]

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = False
        pipeline_options.ocr_options = TesseractCliOcrOptions(
            lang=tesseract_langs,
            tesseract_cmd="/opt/homebrew/bin/tesseract",
            psm=6,
        )

        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )
        result = converter.convert(str(input_path))
        document = getattr(result, "document", None)
    except Exception as exc:
        logger.warning("Docling extraction skipped: %s", exc)
        return None

    if document is None:
        return None

    text = normalize_multiline_text(document.export_to_markdown())
    return text or None


def extract_text_with_pypdf(input_path: Path) -> Optional[str]:
    if input_path.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    try:
        reader = PdfReader(str(input_path))
        pieces: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            normalized = normalize_multiline_text(page_text)
            if normalized:
                pieces.append(normalized)
        text = "\n".join(pieces)
    except Exception as exc:
        logger.warning("PyPDF extraction skipped: %s", exc)
        return None

    return text or None


def choose_primary_text(
    pypdf_text: Optional[str],
    docling_text: Optional[str],
    ocr_text: str,
) -> tuple[str, str]:
    def score(text: str) -> int:
        markers = 0
        lowered = text.casefold()
        for marker in ("article", "chapter", "part", "المادة", "الفصل", "الباب", "الجزء"):
            if marker in lowered:
                markers += 1
        return len(text) + (markers * 200)

    candidates: list[tuple[str, str]] = []
    if pypdf_text:
        candidates.append(("pypdf", pypdf_text))
    if docling_text:
        candidates.append(("docling", docling_text))
    if ocr_text:
        candidates.append(("paddleocr", ocr_text))
    if not candidates:
        return "", "none"
    return max(candidates, key=lambda item: score(item[1]))


def normalize_source_lines(text: str) -> list[str]:
    raw_lines = text.replace("\r", "\n").split("\n")
    cleaned: list[str] = []
    for line in raw_lines:
        normalized = normalize_text(line)
        normalized = re.sub(r"^#{1,6}\s*", "", normalized)
        if re.fullmatch(r"<!--\s*image\s*-->", normalized, flags=re.IGNORECASE):
            continue
        if normalized:
            cleaned.append(normalized)
    return cleaned


def detect_part(line: str) -> Optional[tuple[str, str]]:
    match = PART_RE.match(line)
    if not match:
        arabic_match = re.match(r"^\s*(?:الباب|الجزء)\s+(.+?)\s*$", line, re.IGNORECASE)
        if not arabic_match:
            return None
        extracted = extract_leading_arabic_number_phrase(arabic_match.group(1))
        if not extracted:
            return None
        number, _ = extracted
        return number, normalize_text(line)
    number, remainder = match.groups()
    title = extract_title_from_heading("Part", number, remainder)
    return normalize_digits(number), title


def detect_chapter(line: str) -> Optional[tuple[str, str]]:
    match = CHAPTER_RE.match(line)
    if not match:
        arabic_match = re.match(r"^\s*(?:الفصل)\s+(.+?)\s*$", line, re.IGNORECASE)
        if not arabic_match:
            return None
        extracted = extract_leading_arabic_number_phrase(arabic_match.group(1))
        if not extracted:
            return None
        number, _ = extracted
        return number, normalize_text(line)
    number, remainder = match.groups()
    title = extract_title_from_heading("Chapter", number, remainder)
    return normalize_digits(number), title


def detect_article(line: str) -> Optional[tuple[str, str]]:
    match = ARTICLE_WITH_BODY_RE.match(line)
    if not match:
        match = ARTICLE_STANDALONE_RE.match(line)
        if not match:
            arabic_match = re.match(r"^\s*(?:المادة|مادة)\s+(.+?)\s*$", line, re.IGNORECASE)
            if not arabic_match:
                return None
            extracted = extract_leading_arabic_number_phrase(arabic_match.group(1))
            if not extracted:
                return None
            number, remainder = extracted
            return number, remainder
        number = match.group(1)
        return normalize_digits(number), ""

    number, remainder = match.groups()
    return normalize_digits(number), normalize_text(remainder)


def detect_top_level_chapter(line: str) -> Optional[str]:
    if FINAL_PROVISIONS_RE.match(line):
        return normalize_text(line)
    return None


def is_structure_line(line: str) -> bool:
    return any(
        detector(line) is not None
        for detector in (detect_top_level_chapter, detect_part, detect_chapter, detect_article)
    )


def looks_like_noise(line: str) -> bool:
    text = normalize_text(line)
    if not text:
        return True
    if len(text) <= 2:
        return True
    if not (contains_latin(text) or contains_arabic(text)):
        return True
    letters = sum(char.isalpha() for char in text)
    non_space = sum(not char.isspace() for char in text)
    if non_space and letters / non_space < 0.35:
        return True
    return False


def score_title_candidate(line: str) -> float:
    text = normalize_text(line)
    lowered = text.casefold()
    score = 0.0

    if looks_like_noise(text):
        return -100.0
    if is_structure_line(text):
        return -50.0

    if 8 <= len(text) <= 90:
        score += 8.0
    if len(text.split()) <= 10:
        score += 3.0
    if ":" not in text:
        score += 2.0
    if lowered.endswith("law") or " law" in lowered:
        score += 20.0
    if any(keyword in lowered for keyword in ("chambers", "commerce", "regulation", "regulations", "rules")):
        score += 6.0
    if any(keyword in text for keyword in ("نظام", "قانون", "لائحة", "الأحكام")):
        score += 20.0
    if text == text.title():
        score += 2.0
    if text.endswith(":"):
        score -= 5.0
    if len(text.split()) > 18:
        score -= 8.0
    return score


def select_document_title(source_lines: list[str], preamble_lines: list[str]) -> tuple[Optional[str], Optional[str]]:
    candidates: list[str] = []

    for line in preamble_lines:
        if not looks_like_noise(line):
            candidates.append(line)

    for line in source_lines[:80]:
        normalized = normalize_text(line)
        if looks_like_noise(normalized):
            continue
        if is_structure_line(normalized):
            continue
        candidates.append(normalized)

    if not candidates:
        return None, None

    best_line = max(candidates, key=score_title_candidate)
    if score_title_candidate(best_line) < 0:
        return None, None

    return split_bilingual_text(best_line)


def assign_article(parent_document: dict[str, Any], current_part: Optional[dict[str, Any]], current_chapter: Optional[dict[str, Any]], article: dict[str, Any]) -> None:
    if current_chapter is not None:
        current_chapter["articles"].append(article)
        return
    if current_part is not None:
        current_part["articles"].append(article)
        return
    parent_document["articles"].append(article)


def count_total_articles(document: dict[str, Any]) -> int:
    total = len(document.get("articles", []))
    for chapter in document.get("chapters", []):
        total += len(chapter.get("articles", []))
    for part in document.get("parts", []):
        total += len(part.get("articles", []))
        for chapter in part.get("chapters", []):
            total += len(chapter.get("articles", []))
    return total


def parse_legal_structure(text: str) -> dict[str, Any]:
    document = build_document_template()
    source_lines = normalize_source_lines(text)
    preamble_lines: list[str] = []
    current_part: Optional[dict[str, Any]] = None
    current_chapter: Optional[dict[str, Any]] = None
    current_article: Optional[ArticleContext] = None
    next_order_index = 1
    structure_started = False

    def flush_article() -> None:
        nonlocal current_article
        if current_article is None:
            return
        article_json = current_article.to_json()
        assign_article(document, current_part, current_chapter, article_json)
        current_article = None

    for line in source_lines:
        top_level_chapter_title = detect_top_level_chapter(line)
        part_info = detect_part(line)
        chapter_info = detect_chapter(line)
        article_info = detect_article(line)

        if top_level_chapter_title:
            structure_started = True
            flush_article()
            current_part = None
            current_chapter = build_chapter(top_level_chapter_title, None)
            document["chapters"].append(current_chapter)
            continue

        if part_info:
            structure_started = True
            flush_article()
            current_part = build_part(part_info[1], part_info[0])
            document["parts"].append(current_part)
            current_chapter = None
            continue

        if chapter_info:
            structure_started = True
            flush_article()
            current_chapter = build_chapter(chapter_info[1], chapter_info[0])
            if current_part is not None:
                current_part["chapters"].append(current_chapter)
            else:
                document["chapters"].append(current_chapter)
            continue

        if article_info:
            structure_started = True
            flush_article()
            current_article = ArticleContext(
                article_number=article_info[0],
                order_index=next_order_index,
            )
            next_order_index += 1
            if article_info[1]:
                current_article.lines.append(article_info[1])
            continue

        if current_article is not None:
            current_article.lines.append(line)
            continue

        if not structure_started:
            preamble_lines.append(line)

    flush_article()

    title_en, title_ar = select_document_title(source_lines, preamble_lines)
    document["title_en"] = title_en
    document["title_ar"] = title_ar

    if document["title_en"] is None and document["title_ar"] is None:
        fallback_title_en = "Untitled Legal Document"
        fallback_title_ar = "مستند قانوني بدون عنوان"
        if contains_arabic(text) and not contains_latin(text):
            document["title_ar"] = fallback_title_ar
        else:
            document["title_en"] = fallback_title_en

    return document


def collect_metadata(input_path: Path) -> dict[str, Any]:
    metadata = {
        "file_name": input_path.name,
        "file_path": str(input_path.resolve()),
        "file_size_bytes": input_path.stat().st_size,
        "file_type": "pdf" if input_path.suffix.lower() == ".pdf" else "image",
        "ocr_engine": "PaddleOCR",
        "layout_engine": "Docling",
    }
    if input_path.suffix.lower() == ".pdf":
        try:
            pdf2image = require_dependency("pdf2image", "pip install pdf2image")
            info = pdf2image.pdfinfo_from_path(str(input_path))
            metadata["total_pages"] = int(info["Pages"])
        except Exception:
            metadata["total_pages"] = None
    else:
        metadata["total_pages"] = 1
    return metadata


def run_pipeline(
    file_path: str,
    output_path: Optional[str] = None,
    language_mode: str = "auto",
    disable_docling: bool = False,
) -> dict[str, Any]:
    start_time = time.time()
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    metadata = collect_metadata(input_path)
    pypdf_text = extract_text_with_pypdf(input_path)
    docling_text = None if disable_docling else extract_text_with_docling(input_path, language_mode=language_mode)
    ocr_lines: list[OCRLine] = []
    ocr_text = ""

    text_source, primary_text = choose_primary_text(pypdf_text, docling_text, ocr_text)
    document = parse_legal_structure(primary_text)

    # Legal documents should yield article nodes. If a text extractor returns
    # almost no structure, fall back to OCR instead of saving a misleading JSON.
    if count_total_articles(document) == 0:
        should_retry_with_ocr = (
            text_source in {"docling", "pypdf"}
            or (not pypdf_text and not docling_text)
        )
        if should_retry_with_ocr:
            logger.warning(
                "Primary text source '%s' produced no articles; retrying with PaddleOCR.",
                text_source,
            )
            ocr_lines = run_paddle_ocr(input_path, language_mode=language_mode)
            ocr_text = lines_to_text(ocr_lines)
            text_source, primary_text = choose_primary_text(None, None, ocr_text)
            document = parse_legal_structure(primary_text)

    document["metadata"] = {
        **metadata,
        "processing_time_seconds": round(time.time() - start_time, 2),
        "language_mode": language_mode,
        "pypdf_used": bool(pypdf_text),
        "docling_used": bool(docling_text),
        "ocr_line_count": len(ocr_lines),
        "text_source": text_source,
    }

    if output_path is None:
        output_path = f"{input_path.stem}_structured.json"

    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)

    logger.info("Structured JSON saved to %s", output_file)
    return document


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR legal PDF/image files into structured bilingual JSON."
    )
    parser.add_argument("input", help="Path to the PDF or image file")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument(
        "-l",
        "--lang",
        dest="language_mode",
        choices=["auto", "eng", "ara"],
        default="auto",
        help="OCR language mode",
    )
    parser.add_argument(
        "--no-docling",
        action="store_true",
        help="Disable Docling extraction and rely only on PaddleOCR",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run_pipeline(
            file_path=args.input,
            output_path=args.output,
            language_mode=args.language_mode,
            disable_docling=args.no_docling,
        )
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
